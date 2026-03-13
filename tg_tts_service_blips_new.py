import io
import json
import logging
import os
import time
from typing import *

import torch
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tts.blips")

from torchaudio._extension.utils import _init_dll_path

_init_dll_path()  # I LOVE PYTORCH I LOVE PYTORCH I LOVE PYTORCH FUCKING TORCHAUDIO SUCKS ASS
import asyncio
import io
import json
import os
import random
import re
import threading

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_qwen3_tts import FasterQwen3TTS
from flask import Flask, request, send_file
from pydub import AudioSegment, effects
from tqdm import tqdm

voice_name_mapping = {}
sfx_sound_mapping = {}
use_voice_name_mapping = True
with open("./voice_mapping.json", "r") as file:
    voice_name_mapping = json.load(file)
    if len(voice_name_mapping) == 0:
        use_voice_name_mapping = False
with open("./sfx_mapping.json", "r") as file:
    sfx_sound_mapping = json.load(file)


def audiosegment_to_numpy(seg):
    samples = np.array(seg.get_array_of_samples())

    if seg.channels == 2:
        samples = samples.reshape((-1, 2))

    samples = samples.astype(np.float32) / (1 << (8 * seg.sample_width - 1))

    return samples, seg.frame_rate


def numpy_to_audiosegment(samples, sr, sample_width=2, channels=1):
    samples_int16 = (samples * 32767).astype(np.int16)

    return AudioSegment(
        samples_int16.tobytes(),
        frame_rate=sr,
        sample_width=sample_width,
        channels=channels,
    )


app = Flask(__name__)
letters_to_use = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
sfx_to_use = "&@\{\}[]()^.*!?\\/#~-%_><"
random_factor = 0.35
os.makedirs("samples", exist_ok=True)
trim_leading_silence = lambda x: x[detect_leading_silence(x) :]
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))
global request_count
blips_cache = {}
import math

import pydub.effects


def change_volume(seg, multiplier):
    return seg.apply_gain(20 * math.log10(multiplier))


def normalize_to_target(seg, target_dbfs=-20.0):
    change = target_dbfs - seg.dBFS
    return seg.apply_gain(change)


def cap_loudness(seg, max_dbfs=-1.0):
    if seg.max_dBFS > max_dbfs:
        change = max_dbfs - seg.max_dBFS
        return seg.apply_gain(change)
    return seg


@app.route("/generate-tts-blips")
def text_to_speech_blips():
    global blips_cache
    text = request.json.get("text", "")
    voice = request.json.get("voice", "")
    blip_base = request.json.get("blip_base", "")
    blip_number = request.json.get("blip_number", "")
    pitch = request.json.get("pitch", "")
    request_start_time = time.time()
    logger.debug(
        f"Endpoint: /generate-tts-blips | Voice: {voice} | Base: {blip_base} | Number: {blip_number} | Pitch: {pitch} | Text: {text[:50]}..."
    )
    if pitch == "":
        pitch = "0"
    # print(voice + " blips, " + "\"" + text + "\"")
    if use_voice_name_mapping:
        voice = voice_name_mapping[voice]
    result = None
    actual_text_found = False
    skip_these = " ,:;'\""
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
            result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
            return result
        with torch.no_grad():
            result_sound = AudioSegment.empty()
            if not voice in blips_cache:
                logger.debug(f"Cache miss for blip voice: {voice}. Loading from disk.")
                blips_cache[voice] = torch.load(
                    "./speaker_latents/" + voice + ".blips", weights_only=False
                )
            else:
                logger.debug(f"Cache hit for blip voice: {voice}")

            gen_start = time.time()
            for i, letter in enumerate(text):
                if not letter.isalpha() and not letter.isnumeric():
                    if letter in skip_these:
                        continue
                        # letter_sound = AudioSegment.empty()
                        # new_sound = letter_sound._spawn(b'\x00' * (48000 // 3), overrides={'frame_rate': 48000})
                        # new_sound = new_sound.set_frame_rate(48000)
                        # if not i == 0:
                        # 	result_sound = result_sound.append(new_sound, crossfade = 50)
                        # else:
                        # 	result_sound = new_sound
                    else:
                        if letter == " ":
                            # letter_sound = AudioSegment.empty()
                            # new_sound = letter_sound._spawn(b'\x00' * (48000 // 1.5), overrides={'frame_rate': 48000})
                            # new_sound = new_sound.set_frame_rate(48000)
                            # if not i == 0:
                            # 	result_sound = result_sound.append(new_sound, crossfade = 50)
                            # else:
                            # 	result_sound = new_sound
                            continue
                        if letter == "?" or letter == "!":
                            if not i == len(text) - 1:
                                continue
                        path = "default"
                        if letter in sfx_to_use:
                            path = sfx_sound_mapping[letter]
                        file_path = "blips_sfx/" + path + ".wav"

                        letter_sound = AudioSegment.from_file(file_path)
                        samples, sr = audiosegment_to_numpy(letter_sound)
                        new_audio = numpy_to_audiosegment(
                            samples,
                            sr,
                            sample_width=letter_sound.sample_width,
                            channels=letter_sound.channels,
                        )
                        new_audio = change_volume(new_audio, 0.3)
                        if letter == "?" or letter == "!":
                            letter_sound = AudioSegment.from_file(
                                io.BytesIO(
                                    blips_cache[voice][blip_base][str(blip_number)][
                                        "Deska" if letter == "?" else "Gwah"
                                    ].getvalue()
                                ),
                                format="wav",
                            )
                            samples, sr = audiosegment_to_numpy(letter_sound)
                            detune = 0
                            base_pitch = 0
                            random_pitch = 0.2
                            base_var = 0.2

                            detune = ((0 + base_pitch) * 100) + (
                                (random.random() * (300 + 300) - 300)
                                * (base_var + random_pitch)
                            )

                            semitones = detune / 100
                            # print(semitones)
                            if semitones != 0:
                                samples = librosa.effects.pitch_shift(
                                    samples, sr=sr, n_steps=semitones
                                )

                            # stretched = librosa.effects.time_stretch(samples, rate=2)
                            speech_audio = numpy_to_audiosegment(
                                samples,
                                sr,
                                sample_width=letter_sound.sample_width,
                                channels=letter_sound.channels,
                            )
                            speech_audio = change_volume(speech_audio, 0.6)
                            stripped_sound = strip_silence(speech_audio)
                            new_audio = new_audio.overlay(stripped_sound)
                            # print("ran shit")
                        if not i == 0:
                            result_sound = result_sound.append(new_audio, crossfade=150)
                        else:
                            result_sound = new_audio
                else:
                    if not i % 2 == 0:
                        continue  # Skip every other letter

                    letter_sound = AudioSegment.from_file(
                        io.BytesIO(
                            blips_cache[voice][blip_base][str(blip_number)][
                                letter.lower()
                            ].getvalue()
                        ),
                        format="wav",
                    )
                    # print(letter_sound.duration_seconds)
                    # new_sound = letter_sound._spawn(letter_sound.raw_data, overrides={
                    # 	"frame_rate": int(letter_sound.frame_rate * 1.5)
                    # })
                    samples, sr = audiosegment_to_numpy(letter_sound)
                    detune = 0
                    base_pitch = 1.6 if letter.isupper() else 0
                    random_pitch = 0.15 if letter.isupper() else 0
                    base_var = 0.2

                    detune = ((0 + base_pitch) * 100) + (
                        (random.random() * (300 + 300) - 300)
                        * (base_var + random_pitch)
                    )

                    semitones = detune / 100
                    # print(semitones)
                    if semitones != 0:
                        samples = librosa.effects.pitch_shift(
                            samples, sr=sr, n_steps=semitones
                        )
                    if pitch != "0":
                        samples = librosa.effects.pitch_shift(
                            samples, sr=sr, n_steps=int(pitch), bins_per_octave=24
                        )
                    # stretched = librosa.effects.time_stretch(samples, rate=2)
                    new_audio = numpy_to_audiosegment(
                        samples,
                        sr,
                        sample_width=letter_sound.sample_width,
                        channels=letter_sound.channels,
                    )
                    new_audio = change_volume(
                        new_audio, 0.7 if letter.isupper() else 0.5
                    )
                    stripped_sound = strip_silence(new_audio)
                    # raw = stripped_sound.raw_data[10000:-15000]
                    # octaves = 1 + random.random() * random_factor
                    # frame_rate = int(stripped_sound.frame_rate * (2.0 ** octaves))

                    # new_sound = stripped_sound._spawn(raw, overrides={'frame_rate': frame_rate})
                    # new_sound = new_sound.set_frame_rate(48000)
                    if not i == 0:
                        result_sound = result_sound.append(
                            new_audio.fade_in(3).fade_out(8), crossfade=150
                        )
                    else:
                        result_sound = new_audio.fade_in(3).fade_out(8)

            logger.info(f"Blip synthesis loop took: {time.time() - gen_start:.4f}s")
            result_sound.export(data_bytes, format="wav")
            rawsound = AudioSegment.from_file(io.BytesIO(data_bytes.getvalue()), "wav")
            normalizedsound = normalize_to_target(rawsound, -25)
            normalizedsound = cap_loudness(normalizedsound, max_dbfs=-5)
            # normalizedsound = effects.normalize(rawsound, headroom=1.0)
            normalizedsound.export(data_bytes, format="wav")

        result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")

    logger.info(f"Total processing time: {time.time() - request_start_time:.4f}s")
    return result


@app.route("/tts-voices")
def voices_list():
    if use_voice_name_mapping:
        data = list(voice_name_mapping.keys())
        data.sort()
        return json.dumps(data)


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
                json.dumps(
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
    return json.dumps({"status": "success", "new_level": level_name})


if __name__ == "__main__":
    from waitress import serve

    print("Beginning voice caching")
    for voice, latent in tqdm(voice_name_mapping.items()):
        if not os.path.exists("./speaker_latents/" + latent + ".blips"):
            print("No " + latent + " path found in blips for " + voice)
            continue

        # blips_cache[voice] = torch.load("./speaker_latents/" + voice + ".blips", weights_only=False)

    print("Cached voices.")
    print("Serving TTS Blips on :5004")
    serve(app, host="0.0.0.0", port=5004, backlog=32, channel_timeout=8)
