
import json
import random

from huggingface_hub import snapshot_download
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

from nanovllm_voxcpm import VoxCPM

from qwen_asr import Qwen3ASRModel

asr_model = Qwen3ASRModel.from_pretrained(
	"Qwen/Qwen3-ASR-1.7B",
	dtype=torch.bfloat16,
	device_map="cuda:0",
	attn_implementation="flash_attention_2",
	max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
	max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)
lines = ["We're smokin' filtered crack, you stupid piece of shit, I'll fuckin' kill you!",
		 "I was flippin' bricks for Mansa Musa before y'all even became a type-1 civilization!",
		 "This shit ain't nothin' to me, man!"]
from typing import Any, Generator, List, Optional, Tuple, Union

root_folder = "/mnt/e/hfc_en-US_F/hifi_captain_voices"
import io 

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
				if not os.path.isfile("./speaker_latents/" + prompt_id + ".speaker_latent"):
					print("Speaker " + prompt_id + " not cached, caching...")
					audio_extensions = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
					from tqdm import tqdm
					# Recursively find all audio files
					audio_files = []
					for dirpath, _, filenames in os.walk(root_folder + "/" + prompt_id):
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
					prompt_latents = await self.encode_latents(data_bytes.getvalue(), "wav")
					ref_audio_latents = prompt_latents
					self._prompt_pool[prompt_id] = {"latents": prompt_latents, "text": ref_text}
					torch.save({"latents": prompt_latents, "text": ref_text}, "./speaker_latents/" + prompt_id + ".speaker_latent")
				else:
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
if __name__ == "__main__":
	model = VoxCPM2_tg.from_pretrained(
		"openbmb/VoxCPM2",
		devices=[0],
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.95,
	)
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
				line_start_time = time.perf_counter()
				chunks = [chunk for chunk in model.generate(target_text=line, prompt_id = subfolder.name, max_generate_length = 256)]
				wav = np.concatenate(chunks, axis=0)
				sf.write("./test_lines/" + subfolder.name + "_" + str(current_line) + ".wav", wav, 48000)
				line_end_time = time.perf_counter()
				print("Line generation ended taking " + str(line_end_time - line_start_time) + " seconds.")
	end_time = time.perf_counter()
	print("Generation ended taking " + str(end_time - start_time) + " seconds.")
	model.stop()