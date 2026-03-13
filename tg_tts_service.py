import io
import json
import os
import logging
import time
from typing import *

import torch
from flask import jsonify
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
from faster_qwen3_tts import FasterQwen3TTS
from flask import Flask, request, send_file
from pydub import AudioSegment, effects
from tqdm import tqdm

class Qwen3_TTS_TG(FasterQwen3TTS):

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = torch.bfloat16,
        attn_implementation: str = "sdpa",
        max_seq_len: int = 2048,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("CUDA graphs require CUDA device")

        from faster_qwen3_tts.utils import suppress_flash_attn_warning

        # Import here to avoid dependency issues (and suppress flash-attn warning)
        with suppress_flash_attn_warning():
            from qwen_tts import Qwen3TTSModel
        from faster_qwen3_tts.predictor_graph import PredictorGraph
        from faster_qwen3_tts.talker_graph import TalkerGraph

        # Load base model using qwen-tts library
        base_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        talker = base_model.model.talker
        talker_config = base_model.model.config.talker_config

        # Extract predictor config from loaded model
        predictor = talker.code_predictor
        pred_config = predictor.model.config
        talker_hidden = talker_config.hidden_size
        predictor_graph = PredictorGraph(
            predictor,
            pred_config,
            talker_hidden,
            device=device,
            dtype=dtype,
            do_sample=True,
            top_k=50,
            temperature=0.9,
        )
        talker_graph = TalkerGraph(
            talker.model,
            talker_config,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )
        instance = cls(
            base_model=base_model,
            predictor_graph=predictor_graph,
            talker_graph=talker_graph,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )
        instance._compile_codec(True)
        return instance

    def _compile_codec(self, mode: Union[bool, str] = True) -> None:
        """Apply ``torch.compile`` to the speech tokenizer codec for faster decoding.

        The codec decoder contains 100+ attention modules that benefit greatly
        from compilation, as it eliminates per-module Python dispatch overhead.
        Profiling shows the codec accounts for ~47% of single-generation time
        and ~85% of batch generation time.  Compilation can improve batch
        throughput by 3–4x.

        Args:
                mode:
                        ``True`` to compile with ``mode="max-autotune"``, or a
                        ``torch.compile`` mode string (``"max-autotune"``,
                        ``"reduce-overhead"``, ``"default"``).
        """
        compile_mode = "max-autotune" if mode is True else str(mode)
        codec = self.model.model.speech_tokenizer.model
        self.model.model.speech_tokenizer.model = torch.compile(
            codec,
            mode=compile_mode,
            dynamic=True,
        )

    def _prepare_generation_tg(
        self,
        text: str,
        ref_speaker: str,
    ):
        """Prepare inputs for generation (shared by streaming and non-streaming).

        Args:
                xvec_only: When True (default), use only the speaker embedding (x-vector) for voice
                        cloning instead of the full ICL acoustic prompt. This prevents the model from
                        continuing the reference audio's last phoneme and allows natural language switching.
                        When False, the full reference audio codec tokens are included in context (ICL mode).
        """
        input_texts = [self.model._build_assistant_text(text)]
        input_ids = self.model._tokenize_texts(input_texts)
        if not self.latent_cache:
            self.latent_cache = {}

        vcp, ref_ids = self.latent_cache[ref_speaker]

        m = self.model.model

        tie, tam, tth, tpe = self._build_talker_inputs_local(
            m=m,
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=vcp,
            languages=["English"],
            speakers=None,
            non_streaming_mode=True,
        )

        if not self._warmed_up:
            self._warmup(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None

        ref_codes = vcp["ref_code"][0]

        return m, talker, config, tie, tam, tth, tpe, ref_codes

    @torch.inference_mode()
    def generate_voice_clone_tg(
        self,
        text: str,
        ref_speaker: str,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        from faster_qwen3_tts.generate import fast_generate

        m, talker, config, tie, tam, tth, tpe, ref_codes = self._prepare_generation_tg(
            text,
            ref_speaker,
        )
        codec_ids, _ = fast_generate(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=self.predictor_graph,
            talker_graph=self.talker_graph,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        if codec_ids is None:
            # print("Generation returned no tokens")
            return [np.zeros(1, dtype=np.float32)], self.sample_rate

        # In ICL mode: prepend reference codes before decoding so the codec decoder
        # has acoustic context from the reference audio (matches official implementation).
        speech_tokenizer = m.speech_tokenizer
        if ref_codes is not None:
            ref_codes_dev = ref_codes.to(codec_ids.device)
            codes_for_decode = torch.cat([ref_codes_dev, codec_ids], dim=0)
        else:
            codes_for_decode = codec_ids
        audio_list, sr = speech_tokenizer.decode(
            {"audio_codes": codes_for_decode.unsqueeze(0)}
        )

        # Convert to numpy and trim off the reference audio portion
        ref_len = ref_codes.shape[0] if ref_codes is not None else 0
        total_len = codes_for_decode.shape[0]
        audio_arrays = []
        for a in audio_list:
            if hasattr(a, "cpu"):  # torch tensor
                a = a.flatten().cpu().numpy()
            else:  # already numpy
                a = a.flatten() if hasattr(a, "flatten") else a
            if ref_len > 0:
                cut = int(ref_len / max(total_len, 1) * len(a))
                a = a[cut:]
            audio_arrays.append(a)
        return audio_arrays, sr


import io

print("I: Loading Qwen3-TTS and LavaSR into memory...")
model = Qwen3_TTS_TG.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
from LavaSR.model import LavaEnhance2

lava_model = LavaEnhance2("YatharthS/LavaSR", "cuda")
tts_lock = threading.Lock()
print("Done loading.")
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
    with tts_lock:
        with io.BytesIO() as data_bytes:
            for i, letter in enumerate(text):
                if letter in letters_to_use:
                    actual_text_found = True
                    break
            if not actual_text_found:
                logger.debug("No alphanumeric characters found in text, returning stub file.")
                stub_file = AudioSegment.empty()
                stub_file.set_frame_rate(48000)
                stub_file.export(data_bytes, format="wav")
                result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
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
                audio_list, sr = model.generate_voice_clone_tg(text=text, ref_speaker=voice)
                gen_end = time.time()
                logger.info(f"Voice generation time: {gen_end - gen_start:.4f}s")

                sf.write(data_bytes, audio_list[0], sr, format="wav")
                
                enhance_start = time.time()
                input_audio, _ = lava_model.load_audio(
                    io.BytesIO(data_bytes.getvalue()), input_sr=24000
                )
                output_audio = (
                    lava_model.enhance(input_audio, denoise=False).cpu().numpy().squeeze()
                )
                enhance_end = time.time()
                logger.info(f"LavaSR enhancement time: {enhance_end - enhance_start:.4f}s")

                temp_databytes = io.BytesIO()
                sf.write(temp_databytes, output_audio, 48000, format="wav")
                rawsound = AudioSegment.from_file(
                    io.BytesIO(temp_databytes.getvalue()), "wav"
                )
                normalizedsound = normalize_to_target(rawsound, -25)
                normalizedsound = cap_loudness(normalizedsound, max_dbfs=-5)
                normalizedsound = effects.normalize(rawsound)
                normalizedsound.export(temp_databytes, format="wav")
                result = send_file(
                    io.BytesIO(temp_databytes.getvalue()), mimetype="audio/wav"
                )
        
        logger.info(f"Total processing time: {time.time() - request_start_time:.4f}s")
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
    current_level = logger.getEffectiveLevel()
    new_level = logging.DEBUG if current_level == logging.INFO else logging.INFO
    logger.setLevel(new_level)
    # Also update the base logger if needed, but updating 'logger' is usually enough for this file
    level_name = logging.getLevelName(new_level)
    return jsonify({"status": "success", "new_level": level_name})


if __name__ == "__main__":
    from waitress import serve

    print("Beginning voice caching")
    model.latent_cache = {}
    for k, v in tqdm(voice_name_mapping.items()):
        model.latent_cache[k] = torch.load("./speaker_latents/" + v + ".speaker_latent")
    print("Cached voices.")
    print("Warming model up...")
    with tts_lock:
        trash = io.BytesIO()
        audio_list, sr = model.generate_voice_clone_tg(
            text="The quick brown fox jumps over the lazy dog.",
            ref_speaker=list(voice_name_mapping)[0],
        )
        sf.write(trash, audio_list[0], sr, format="wav")
        input_audio, _ = lava_model.load_audio(
            io.BytesIO(trash.getvalue()), input_sr=24000
        )
        output_audio = (
            lava_model.enhance(input_audio, denoise=False).cpu().numpy().squeeze()
        )
        del trash
        del input_audio
        del output_audio
        del _
    print("Serving TTS on :5003")
    serve(app, host="0.0.0.0", port=5003, backlog=32, channel_timeout=8)
