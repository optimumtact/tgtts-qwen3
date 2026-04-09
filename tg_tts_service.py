import io
import json
import logging
import os
import time
from typing import *

from huggingface_hub import snapshot_download
import torch
from flask import jsonify
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tts.service")

from torchaudio._extension.utils import _init_dll_path

_init_dll_path()  # I LOVE PYTORCH I LOVE PYTORCH I LOVE PYTORCH FUCKING TORCHAUDIO SUCKS ASS
import asyncio
import io
import json
import os
import random
import re
import threading
import time

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from flask import Flask, request, send_file
from pydub import AudioSegment, effects
from tqdm import tqdm
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm2.server import SyncVoxCPM2ServerPool, AsyncVoxCPM2ServerPool, AsyncVoxCPM2Server
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

class TGAsyncPool(AsyncVoxCPM2ServerPool):
    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        prompt_id: str | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
        ref_audio_latents: bytes | None = None,
    ):
        if prompt_id is not None:
            if prompt_id not in self._prompt_pool:
                self._prompt_pool[prompt_id] = torch.load("./speaker_latents/" + prompt_id + ".speaker_latent", weights_only=False)

            prompt_info = self._prompt_pool[prompt_id]
            prompt_latents = prompt_info["latents"]
            prompt_text = prompt_info["text"]
            ref_audio_latents = prompt_latents

        min_load_server_idx = np.argmin(self.servers_load)
        self.servers_load[min_load_server_idx] += 1
        server = self.servers[min_load_server_idx]
        try:
            async for data in server.generate(
                target_text,
                prompt_latents,
                prompt_text,
                max_generate_length,
                temperature,
                cfg_value,
                ref_audio_latents,
            ):
                yield data
        finally:
            self.servers_load[min_load_server_idx] -= 1

class TGServerPool(SyncVoxCPM2ServerPool):
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: List[int] = [],
        lora_config: Optional[LoRAConfig] = None,
        **kwargs,
    ):
        async def init_async_server_pool():
            return TGAsyncPool(
                model_path=model_path,
                inference_timesteps=inference_timesteps,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                devices=devices,
                lora_config=lora_config,
                **kwargs,
            )

        self.loop = asyncio.new_event_loop()
        self.server_pool = self.loop.run_until_complete(init_async_server_pool())
        self.loop.run_until_complete(self.server_pool.wait_for_ready())

class VoxCPM2_tg(VoxCPM):
    @staticmethod
    def from_pretrained(
        model: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: List[int] = [],
        lora_config: Any = None,
        **kwargs,
    ):
        if "~" in model:
            model_path = os.path.expanduser(model)
            if not os.path.isdir(model_path):
                raise ValueError(f"Model path {model_path} does not exist")
        else:
            if not os.path.isdir(model):
                model_path = snapshot_download(repo_id=model)
            else:
                model_path = model

        config_file = os.path.expanduser(os.path.join(model_path, "config.json"))

        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file `{config_file}` not found")

        config = json.load(open(config_file))

        arch = config["architecture"]

        if len(devices) == 0:
            devices = [0]

        sync_server_pool_cls = TGServerPool
        return sync_server_pool_cls(
            model_path=model_path,
            inference_timesteps=inference_timesteps,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            devices=devices,
            lora_config=lora_config,
            **kwargs,
        )



import io


tts_lock = threading.Lock()
voice_name_mapping = {}
use_voice_name_mapping = True
with open("./voice_mapping.json", "r") as file:
    voice_name_mapping = json.load(file)
    print("loaded voice mappings")
    if len(voice_name_mapping) == 0:
        use_voice_name_mapping = False
print("voice mappings to use: " + ", ".join(list(voice_name_mapping.values())))
app = Flask(__name__)
letters_to_use = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
random_factor = 0.35
os.makedirs("samples", exist_ok=True)
trim_leading_silence = lambda x: x[detect_leading_silence(x) :]
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))

global request_count


def normalize_to_target(seg, target_dbfs=-20.0):
    change = target_dbfs - seg.dBFS
    return seg.apply_gain(change)


def cap_loudness(seg, max_dbfs=-1.0):
    if seg.max_dBFS > max_dbfs:
        change = max_dbfs - seg.max_dBFS
        return seg.apply_gain(change)
    return seg


@app.route("/generate-tts")
def text_to_speech():
    text = request.json.get("text", "")
    voice = request.json.get("voice", "")
    request_start_time = time.time()
    logger.debug(f"Endpoint: /generate-tts | Voice: {voice} | Text: {text[:50]}...")
    result = None
    actual_text_found = False
    audio_duration = 0
    tts_duration = 0
    lava_duration = 0
    normalize_duration = 0
    with tts_lock:
        with io.BytesIO() as data_bytes:
            for i, letter in enumerate(text):
                if letter in letters_to_use:
                    actual_text_found = True
                    break
            if not actual_text_found:
                logger.debug(
                    "No alphanumeric characters found in text, returning stub file."
                )
                stub_file = AudioSegment.empty()
                stub_file.set_frame_rate(48000)
                stub_file.export(data_bytes, format="wav")
                result = send_file(
                    io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav"
                )
                return result
            with torch.inference_mode():
                final_letter = text[-1]
                acceptable_punctuation = [".", "?", "!"]
                if not final_letter in acceptable_punctuation:
                    text += ". "
                if (
                    text and text[0].isalpha() and not text[0].isupper()
                ):  # capitalize that shit if they forgot
                    text = text[0].upper() + text[1:]

                # Inference
                gen_start = time.time()
                chunks = [chunk for chunk in model.generate(target_text=text, prompt_id = voice_name_mapping[voice], max_generate_length = 256)]
                wav = np.concatenate(chunks, axis=0)
                tts_databytes = io.BytesIO()
                sf.write(tts_databytes, wav, 48000, format="wav")
                gen_end = time.time()
                tts_duration = gen_end - gen_start
                normalise_start = time.time()
                rawsound = AudioSegment.from_file(
                    io.BytesIO(tts_databytes.getvalue()), "wav"
                )
                temp_databytes = io.BytesIO()
                normalizedsound = normalize_to_target(rawsound, -25)
                normalizedsound = cap_loudness(normalizedsound, max_dbfs=-5)
                normalizedsound = effects.normalize(rawsound)
                normalizedsound.export(temp_databytes, format="wav")
                normalised_end = time.time()
                normalize_duration = normalised_end - normalise_start
                result = send_file(
                    io.BytesIO(temp_databytes.getvalue()), mimetype="audio/wav"
                )
                audio_duration = len(normalizedsound) / 1000

        totaltime = time.time() - request_start_time
        if totaltime > 4.0:
            logger.warning(
                f"Slow request detected. Total time: {totaltime:.4f}s | Voice: {voice} | Text: {text} | Audio Duration: {audio_duration:.2f}s | TTS Time: {tts_duration:.4f}s | Normalization Time: {normalize_duration:.4f}s"
            )

        else:
            logger.info(
                f"Request complete Total time: {totaltime:.4f}s | Voice: {voice} | Text: {text} | Audio Duration: {audio_duration:.2f}s | TTS Time: {tts_duration:.4f}s | Normalization Time: {normalize_duration:.4f}s"
            )
        return result


@app.route("/tts-voices")
def voices_list():
    if use_voice_name_mapping:
        data = list(voice_name_mapping.keys())
        data.sort()
        return jsonify(data)


@app.route("/health-check")
def tts_health_check():
    return f"OK: 1", 200


@app.route("/toggle-logging")
def toggle_logging():
    level_str = request.args.get("level", "").upper()
    if level_str:
        try:
            logger.setLevel(level_str)
        except (ValueError, TypeError):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Invalid logging level: {level_str}",
                    }
                ),
                400,
            )
    else:
        current_level = logger.getEffectiveLevel()
        new_level = logging.DEBUG if current_level == logging.INFO else logging.INFO
        logger.setLevel(new_level)

    level_name = logging.getLevelName(logger.getEffectiveLevel())
    return jsonify({"status": "success", "new_level": level_name})


if __name__ == "__main__":
    from waitress import serve
    print("I: Loading Qwen3-TTS and LavaSR into memory...")
    model = VoxCPM2_tg.from_pretrained(
        "openbmb/VoxCPM2",
        devices=[0],
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.95,
    )
    print("Done loading.")
    print("Beginning voice caching")
    first_latent = None
    for k, v in tqdm(voice_name_mapping.items()):
        if not first_latent:
            first_latent = v
            break
    print("Warming model up...")
    with tts_lock:
        trash = io.BytesIO()
        chunks = [chunk for chunk in model.generate(target_text="The quick brown fox jumps over the lazy dog.", prompt_id = first_latent, max_generate_length = 256)]
        wav = np.concatenate(chunks, axis=0)
        sf.write(trash, wav, 48000, format="wav")
        del trash
    print("Serving TTS on :5003")
    serve(app, host="0.0.0.0", port=5003, backlog=32, channel_timeout=8)
    print("Closing server...")
    model.stop()
