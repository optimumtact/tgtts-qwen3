import gc
import io
import os
import random
import re
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tts.api")

import librosa
import numpy as np
import pydub
import pysbd
import requests
import soundfile as sf
from blake3 import blake3
from flask import Flask, abort, make_response, request, send_file, jsonify
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from stftpitchshift import StftPitchShift
import torch
app = Flask(__name__)
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


from scipy.signal import butter, lfilter


def bandpass(x, sr, low=300, high=3000, order=4):
    nyq = sr * 0.5
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, x)


def compress(x, threshold=0.2, ratio=4):
    y = x.copy()
    mask = np.abs(y) > threshold
    y[mask] = np.sign(y[mask]) * (threshold + (np.abs(y[mask]) - threshold) / ratio)
    return y


def saturate(x, drive=2.5):
    return np.tanh(drive * x)


def add_radio_noise(x, level=0.004):
    noise = np.random.normal(0, level, len(x))
    return x + noise


def am_modulate(x, sr, depth=0.08, freq=60):
    t = np.arange(len(x)) / sr
    return x * (1 + depth * np.sin(2 * np.pi * freq * t))


def squelch_tail(sr, length=0.15):
    n = int(sr * length)
    noise = np.random.normal(0, 0.02, n)
    fade = np.linspace(1, 0, n)
    return noise * fade


def normalize(x, peak=0.8):
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m * peak
    return x


def ensure_mono(x):
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def load_and_match(path, target_sr):
    audio, sr = sf.read(path)
    audio = ensure_mono(audio)

    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio


def radio_effect(audio, sr):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    click_on = load_and_match("mic_click_on.wav", sr)
    click_off = load_and_match("mic_click_off.wav", sr)
    static = load_and_match("diffstatic.wav", sr)
    speech = librosa.resample(audio, orig_sr=sr, target_sr=8000)
    speech = librosa.resample(speech, orig_sr=8000, target_sr=48000)
    speech = normalize(speech, 0.5)
    speech = bandpass(speech, sr)
    speech = saturate(speech, 2)
    speech = normalize(speech, 0.7)
    static = static[: len(speech)]
    speech = speech + static * 0.2
    output = np.concatenate([click_on, speech, static[: int(sr * 0.1)], click_off])

    output = normalize(output, 0.9)

    return output


@app.route("/radio")
def radio_handler():
    identifier = request.json.get("identifier", "")
    folder = request.json.get("folder", "")
    start_time = time.time()
    logger.debug(f"ID: {identifier} | Applying radio effect to generated audio...")

    timeout = 10
    start = time.time()

    while not os.path.exists("./cache/" + folder + "/" + identifier + ".radio"):
        if time.time() - start > timeout:
            logger.debug(f"ID: {identifier} | Timed out waiting for the input file!")
            abort(408)
        time.sleep(0.05)
    base_audio = pydub.AudioSegment.from_file(io.BytesIO(torch.load("./cache/" + folder + "/" + identifier + ".radio", weights_only=False)), format="ogg")
    samples, sr = audiosegment_to_numpy(base_audio)
    processed = radio_effect(samples, sr)
    export_audio = numpy_to_audiosegment(processed, sr)
    output_bytes = io.BytesIO()
    export_audio.export(output_bytes, format="ogg")
    result = send_file(
        io.BytesIO(output_bytes.getvalue()), mimetype="audio/ogg"
    )
    result.headers["audio-length"] = export_audio.duration_seconds
    logger.info(f"ID: {identifier} | Radio effect generation time: {time.time() - start_time:.4f}s")
    return result

if __name__ == "__main__":
    from waitress import serve
    print("Warming up...")
    noise = radio_effect(np.random.randn(int(48000 * 1)), 48000)
    del noise
    print("Serving Radio Effects on :5005")
    serve(app, host="0.0.0.0", port=5005, backlog=32, channel_timeout=8)