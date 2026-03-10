
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

from faster_qwen3_tts import FasterQwen3TTS
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

root_folder = "F:/rvc_dataset/xtts_ref_wavs_multiple"
import io 
class Qwen3_TTS_TG(FasterQwen3TTS):
	
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
		if not os.path.isfile("./speaker_latents/" + ref_speaker + ".speaker_latent"):
			print("Speaker " + ref_speaker + " not cached, caching...")
			audio_extensions = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
			from tqdm import tqdm
			# Recursively find all audio files
			audio_files = []
			for dirpath, _, filenames in os.walk(root_folder + "/" + ref_speaker):
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
			read_audio, audio_sr = sf.read(io.BytesIO(data_bytes.getvalue()), always_2d=False)
			if read_audio.ndim > 1:
				read_audio = read_audio.mean(axis=1)  # convert to mono
			silence = np.zeros(int(0.5 * audio_sr), dtype=np.float32)
			read_audio = np.concatenate([read_audio, silence])
			ref_audio_input = (read_audio, audio_sr)
			prompt_items = self.model.create_voice_clone_prompt(
				ref_audio=ref_audio_input,
				ref_text=ref_text
			)
			vcp = self.model._prompt_items_to_voice_clone_prompt(prompt_items)

			ref_ids = []
			rt = prompt_items[0].ref_text
			if rt:
				ref_texts = [self.model._build_ref_text(rt)]
				ref_ids.append(self.model._tokenize_texts(ref_texts)[0])
			else:
				ref_ids.append(None)
			torch.save((vcp, ref_ids), "./speaker_latents/" + ref_speaker + ".speaker_latent")
		else:
			vcp, ref_ids = torch.load("./speaker_latents/" + ref_speaker + ".speaker_latent")

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
			print("Generation returned no tokens")
			return [np.zeros(1, dtype=np.float32)], self.sample_rate

		# In ICL mode: prepend reference codes before decoding so the codec decoder
		# has acoustic context from the reference audio (matches official implementation).
		speech_tokenizer = m.speech_tokenizer
		if ref_codes is not None:
			ref_codes_dev = ref_codes.to(codec_ids.device)
			codes_for_decode = torch.cat([ref_codes_dev, codec_ids], dim=0)
		else:
			codes_for_decode = codec_ids
		audio_list, sr = speech_tokenizer.decode({"audio_codes": codes_for_decode.unsqueeze(0)})

		# Convert to numpy and trim off the reference audio portion
		ref_len = ref_codes.shape[0] if ref_codes is not None else 0
		total_len = codes_for_decode.shape[0]
		audio_arrays = []
		for a in audio_list:
			if hasattr(a, 'cpu'):  # torch tensor
				a = a.flatten().cpu().numpy()
			else:  # already numpy
				a = a.flatten() if hasattr(a, 'flatten') else a
			if ref_len > 0:
				cut = int(ref_len / max(total_len, 1) * len(a))
				a = a[cut:]
			audio_arrays.append(a)
		return audio_arrays, sr

model = Qwen3_TTS_TG.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
from pathlib import Path
import time
from LavaSR.model import LavaEnhance2 
lava_model = LavaEnhance2("YatharthS/LavaSR", "cuda")


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
folder = Path("F:/rvc_dataset/xtts_ref_wavs_multiple")
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
			real_list, real_sr = model.generate_voice_clone_tg(
				text=line, 
				ref_speaker=subfolder.name
			)
			sf.write("./test_lines/" + subfolder.name + "_" + str(current_line) + ".wav", real_list[0], real_sr)
			input_audio, input_sr = lava_model.load_audio("./test_lines/" + subfolder.name + "_" + str(current_line) + ".wav", input_sr=24000)
			output_audio = lava_model.enhance(input_audio, denoise=False).cpu().numpy().squeeze()
			sf.write("./test_lines/" + subfolder.name + "_" + str(current_line) + ".wav", output_audio, 48000)
end_time = time.perf_counter()
print("Generation ended taking " + str(end_time - start_time) + " seconds.")