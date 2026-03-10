import torch
import os
import io
import json
from typing import *
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from torchaudio._extension.utils import _init_dll_path
_init_dll_path() # I LOVE PYTORCH I LOVE PYTORCH I LOVE PYTORCH FUCKING TORCHAUDIO SUCKS ASS
import io
from pydub import AudioSegment, effects  
import json
from flask import Flask, request, send_file
from tqdm import tqdm
import threading


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
			codec, mode=compile_mode, dynamic=True,
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
import io 
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
		channels=channels
	)

app = Flask(__name__)
letters_to_use = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
sfx_to_use = "&@\{\}[]()^.*!?\\/#~-%_><"
random_factor = 0.35
os.makedirs('samples', exist_ok=True)
trim_leading_silence = lambda x: x[detect_leading_silence(x) :]
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))
voice_name_mapping_reversed = {v: k for k, v in voice_name_mapping.items()}
global request_count
blips_cache = {}
import pydub.effects
import math
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
	if pitch == "":
		pitch = "0"
	print(voice + " blips, " + "\"" + text + "\"")
	if use_voice_name_mapping:
		voice = voice_name_mapping_reversed[voice]
	result = None
	actual_text_found = False
	skip_these = " ,:;'\""
	with io.BytesIO() as data_bytes:
		for i, letter in enumerate(text):
			if letter in letters_to_use:
				actual_text_found = True
				break
		if not actual_text_found:
			stub_file = AudioSegment.empty()
			stub_file.set_frame_rate(48000)
			stub_file.export(data_bytes, format = "wav")
			result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
			return result
		with torch.no_grad():
			result_sound = AudioSegment.empty()
			if not voice in blips_cache:
				blips_cache[voice] = torch.load("./speaker_latents/" + voice + ".blips", weights_only=False)
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
							channels=letter_sound.channels
						)
						new_audio = change_volume(new_audio, 0.3)
						if letter == "?" or letter == "!":
							letter_sound = AudioSegment.from_file(io.BytesIO(blips_cache[voice][blip_base][str(blip_number)]["Deska" if letter == "?" else "Gwah"].getvalue()), format="wav")
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
							#print(semitones)
							if semitones != 0:
								samples = librosa.effects.pitch_shift(samples, sr=sr, n_steps=semitones)
							
							#stretched = librosa.effects.time_stretch(samples, rate=2)
							speech_audio = numpy_to_audiosegment(
								samples,
								sr,
								sample_width=letter_sound.sample_width,
								channels=letter_sound.channels
							)
							speech_audio = change_volume(speech_audio, 0.6)
							stripped_sound = strip_silence(speech_audio)
							new_audio = new_audio.overlay(stripped_sound)
							#print("ran shit")
						if not i == 0:
							result_sound = result_sound.append(new_audio, crossfade = 150)
						else:
							result_sound = new_audio
				else:
					if not i % 2 == 0:
						continue # Skip every other letter

					letter_sound = AudioSegment.from_file(io.BytesIO(blips_cache[voice][blip_base][str(blip_number)][letter.lower()].getvalue()), format="wav")
					#print(letter_sound.duration_seconds)
					#new_sound = letter_sound._spawn(letter_sound.raw_data, overrides={
					#	"frame_rate": int(letter_sound.frame_rate * 1.5)
					#})
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
					#print(semitones)
					if semitones != 0:
						samples = librosa.effects.pitch_shift(samples, sr=sr, n_steps=semitones)
					if pitch != "0":
						samples = librosa.effects.pitch_shift(samples, sr=sr, n_steps=int(pitch), bins_per_octave=24)
					#stretched = librosa.effects.time_stretch(samples, rate=2)
					new_audio = numpy_to_audiosegment(
						samples,
						sr,
						sample_width=letter_sound.sample_width,
						channels=letter_sound.channels
					)
					new_audio = change_volume(new_audio, 0.7 if letter.isupper() else 0.5)
					stripped_sound = strip_silence(new_audio)
					# raw = stripped_sound.raw_data[10000:-15000]
					# octaves = 1 + random.random() * random_factor
					# frame_rate = int(stripped_sound.frame_rate * (2.0 ** octaves))

					# new_sound = stripped_sound._spawn(raw, overrides={'frame_rate': frame_rate})
					# new_sound = new_sound.set_frame_rate(48000)
					if not i == 0:
						result_sound = result_sound.append(new_audio.fade_in(3).fade_out(8), crossfade = 150)
					else:
						result_sound = new_audio.fade_in(3).fade_out(8)
			result_sound.export(data_bytes, format = "wav")
			rawsound = AudioSegment.from_file(io.BytesIO(data_bytes.getvalue()), "wav")  
			normalizedsound = normalize_to_target(rawsound, -25)
			normalizedsound = cap_loudness(normalizedsound, max_dbfs=-5)
			#normalizedsound = effects.normalize(rawsound, headroom=1.0)
			normalizedsound.export(data_bytes, format="wav")
		
		result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
	return result

@app.route("/tts-voices")
def voices_list():
	if use_voice_name_mapping:
		data = list(voice_name_mapping.values())
		data.sort()
		return json.dumps(data)
	
@app.route("/health-check")
def tts_health_check():
	return f"OK: 1", 200

if __name__ == "__main__":
	from waitress import serve
	print("Beginning voice caching")
	for voice,v in tqdm(voice_name_mapping.items()):
		if not os.path.exists("./speaker_latents/" + voice + ".blips"):
			print("No blips for " + voice)
			continue

		#blips_cache[voice] = torch.load("./speaker_latents/" + voice + ".blips", weights_only=False)

	print("Cached voices.")
	print("Serving TTS Blips on :5004")
	serve(app, host="0.0.0.0", port=5004, backlog=32, channel_timeout=8)

	
