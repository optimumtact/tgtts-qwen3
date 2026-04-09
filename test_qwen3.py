
import random

import torch
import torchaudio
import librosa
import io
import numpy as np
from pydub import AudioSegment
import os
import re
import asyncio
import soundfile as sf

#from modelscope import snapshot_download
#snapshot_download("OpenBMB/VoxCPM2", local_dir='./pretrained_models/VoxCPM2') # specify the local directory to save the model

from voxcpm import VoxCPM

from qwen_asr import Qwen3ASRModel

asr_model = Qwen3ASRModel.from_pretrained(
	"Qwen/Qwen3-ASR-1.7B",
	dtype=torch.bfloat16,
	device_map="cuda:0",
	# attn_implementation="flash_attention_2",
	max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
	max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)
lines = ["We're smokin' filtered crack, you stupid piece of shit, I'll fuckin' kill you!",
		 "I was flippin' bricks for Mansa Musa before y'all even became a type-1 civilization!",
		 "This shit ain't nothin' to me, man!"]
from typing import Generator, Optional, Tuple, Union

root_folder = "/mnt/e/hfc_en-US_F/hifi_captain_voices"
import io 

class VoxCPM2_tg(VoxCPM):
	
	def _generate(
		self,
		text: str,
		speaker_name: str = None,
		cfg_value: float = 2.0,
		inference_timesteps: int = 10,
		min_len: int = 2,
		max_len: int = 4096,
		retry_badcase: bool = True,
		retry_badcase_max_times: int = 3,
		retry_badcase_ratio_threshold: float = 6.0,
		streaming: bool = False,
	) -> Generator[np.ndarray, None, None]:
		"""Synthesize speech for the given text and return a single waveform.

		Args:
			text: Input text to synthesize.
			prompt_wav_path: The speaker's name.
			cfg_value: Guidance scale for the generation model.
			inference_timesteps: Number of inference steps.
			min_len: Minimum audio length.
			max_len: Maximum token length during generation.
			normalize: Whether to run text normalization before generation.
			denoise: Whether to denoise the prompt/reference audio if a
				denoiser is available.
			retry_badcase: Whether to retry badcase.
			retry_badcase_max_times: Maximum number of times to retry badcase.
			retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
			streaming: Whether to return a generator of audio chunks.
		Returns:
			Generator of numpy.ndarray: 1D waveform array (float32) on CPU.
			Yields audio chunks for each generation step if ``streaming=True``,
			otherwise yields a single array containing the final audio.
		"""
		if not text.strip() or not isinstance(text, str):
			raise ValueError("target text must be a non-empty string")

		text = text.replace("\n", " ")
		text = re.sub(r"\s+", " ", text)
		fixed_prompt_cache = None
		if not os.path.isfile("./speaker_latents/" + speaker_name + ".speaker_latent"):
			print("Speaker " + speaker_name + " not cached, caching...")
			audio_extensions = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
			from tqdm import tqdm
			# Recursively find all audio files
			audio_files = []
			for dirpath, _, filenames in os.walk(root_folder + "/" + speaker_name):
				for f in filenames:
					if f.lower().endswith(audio_extensions):
						audio_files.append(os.path.join(dirpath, f))

			audio_files.sort()

			if not audio_files:
				raise Exception("No audio files found in the folder!")

			combined = AudioSegment.silent(duration=0)
			current_duration = 0
			for file in tqdm(audio_files, desc="Processing", unit="lines"):
				if current_duration >= 30:
					break
				data, samplerate = sf.read(file, dtype="int16")

				if data.ndim == 1:
					data = data[:, np.newaxis]
				audio = AudioSegment(
					data.tobytes(),
					frame_rate=samplerate,
					sample_width=data.dtype.itemsize,
					channels=data.shape[1]
				)
				if audio.duration_seconds < 2:
					continue
				if current_duration + audio.duration_seconds >= 30:
					#print("Skipping " + file + " for duration, current is " + str(current_duration) + ", combined would be " + str(audio.duration_seconds))
					continue
				try:
					output = asr_model.transcribe(
						audio=file,
						language="English", # set "English" to force the language
					)
					lines.append(output[0].text)
				except:
					#print("Skipping " + file + " for failure to transcribe.")
					continue
				combined += audio
				current_duration += audio.duration_seconds
			data_bytes = io.BytesIO()
			combined.export(data_bytes, format="wav")
			combined.export("temp.wav", format="wav")
			output = asr_model.transcribe(
				audio="temp.wav",
				language="English", # set "English" to force the language
			)
			ref_text = output[0].text

			fixed_prompt_cache = self.tts_model.build_prompt_cache(
				prompt_text=ref_text,
				prompt_wav_path="temp.wav",
				reference_wav_path="temp.wav",
			)
			torch.save(fixed_prompt_cache, "./speaker_latents/" + speaker_name + ".speaker_latent")
		else:
			fixed_prompt_cache = torch.load("./speaker_latents/" + speaker_name + ".speaker_latent")

		generate_result = self.tts_model._generate_with_prompt_cache(
			target_text=text,
			prompt_cache=fixed_prompt_cache,
			min_len=min_len,
			max_len=max_len,
			inference_timesteps=inference_timesteps,
			cfg_value=cfg_value,
			retry_badcase=retry_badcase,
			retry_badcase_max_times=retry_badcase_max_times,
			retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
			streaming=streaming,
		)

		for wav, _, _ in generate_result:
			yield wav.squeeze(0).cpu().numpy()


model = VoxCPM2_tg.from_pretrained(
  "openbmb/VoxCPM2",
  load_denoiser=False,
)

from pathlib import Path
import time

from scipy.signal import butter, lfilter
def bandpass(x, sr, low=300, high=3000, order=4):
	nyq = sr * 0.5
	b, a = butter(order, [low/nyq, high/nyq], btype='band')
	return lfilter(b, a, x)

def compress(x, threshold=0.2, ratio=4):
	y = x.copy()
	mask = np.abs(y) > threshold
	y[mask] = np.sign(y[mask]) * (
		threshold + (np.abs(y[mask]) - threshold) / ratio
	)
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

def old():
	audio = normalize(audio, 0.6)
	audio = compress(audio, threshold=0.15, ratio=5)
	audio = bandpass(audio, sr, 300, 3000)
	audio = saturate(audio, drive=2)
	audio = am_modulate(audio, sr)
	audio = add_radio_noise(audio, 0.004)
	audio = normalize(audio, 0.8)
	tail = squelch_tail(sr)
	audio = np.concatenate([audio, tail])

def police_radio_effect(audio, sr):

	if audio.ndim > 1:
		audio = audio.mean(axis=1)

	click_on = load_and_match("mic_click_on.wav", sr)
	click_off = load_and_match("mic_click_off.wav", sr)
	static = load_and_match("diffstatic.wav", sr)
	speech = librosa.resample(audio, orig_sr=sr, target_sr=8000)
	speech = librosa.resample(speech, orig_sr=8000, target_sr=48000)
	# --- radio processing ---
	speech = normalize(speech, 0.5)
	speech = bandpass(speech, sr)
	speech = saturate(speech, 2)
	speech = normalize(speech, 0.7)

	# optional: overlay some static during speech
	static = static[:len(speech)]
	speech = speech + static * 0.2

	# --- assemble transmission ---
	output = np.concatenate([
		click_on,
		speech,
		static[:int(sr*0.1)],  # small tail burst
		click_off
	])

	output = normalize(output, 0.9)

	return output


#sf.write("processed.wav", police_radio_effect(output_audio, 48000), 48000)

ignore = []
print("Begin latent generation:")
from pathlib import Path
start_time = time.perf_counter()
folder = Path("/mnt/e/hfc_en-US_F/hifi_captain_voices")
import tqdm
for subfolder in tqdm.tqdm(folder.iterdir()):
	if subfolder.is_dir():
		if subfolder.name in ignore:
			continue
		#if os.path.isfile("./speaker_latents/" + subfolder.name + ".speaker_latent"):
			#continue
		print(subfolder.name)
		current_line = 0
		for line in random.sample(lines, 3):
			current_line += 1
			wav = model.generate(
				text=line, 
				speaker_name=subfolder.name,
				max_len=128,
				retry_badcase=False
			)
			sf.write("./test_lines/" + subfolder.name + "_" + str(current_line) + ".wav", wav, 48000)
end_time = time.perf_counter()
print("Generation ended taking " + str(end_time - start_time) + " seconds.")