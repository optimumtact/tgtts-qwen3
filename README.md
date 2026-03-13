This is run with docker compose, the tts api relies on the haproxy endpoints and is not configurable

Make sure you have 
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

And you have 
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then run

`docker compose up -d` in the root of the directory.

This will take ages as the pytorch docker image is huge when built.


You can monitor progress via
`docker compose logs tts1 -f`

The output looks like this
```bash
tts1-1  | 
tts1-1  | ==========
tts1-1  | == CUDA ==
tts1-1  | ==========
tts1-1  | 
tts1-1  | CUDA Version 12.6.3
tts1-1  | 
tts1-1  | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
tts1-1  | 
tts1-1  | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
tts1-1  | By pulling and using the container, you accept the terms and conditions of this license:
tts1-1  | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
tts1-1  | 
tts1-1  | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
tts1-1  | 
tts1-1  | I: Loading Qwen3-TTS and LavaSR into memory...
tts1-1  | 
tts1-1  | `torch_dtype` is deprecated! Use `dtype` instead!
Fetching 4 files: 100%|██████████| 4/4 [00:00<00:00, 112598.77it/s]
Fetching 9 files: 100%|██████████| 9/9 [00:00<00:00, 204047.22it/s]
tts1-1  | Done loading.
tts1-1  | loaded voice mappings
tts1-1  | voice mappings to use: hifi_captain_woman, hifi_captain_man
tts1-1  | Beginning voice caching
100%|██████████| 2/2 [00:00<00:00, 1682.10it/s]
tts1-1  | Cached voices.
tts1-1  | Warming model up...
tts1-1  | Warming up predictor (3 runs)...
tts1-1  | Capturing CUDA graph for predictor...
tts1-1  | CUDA graph captured!
tts1-1  | Warming up talker graph (3 runs)...
tts1-1  | Capturing CUDA graph for talker decode...
tts1-1  | Talker CUDA graph captured!
tts1-1  | /opt/conda/lib/python3.11/site-packages/LavaSR/enhancer/enhancer.py:58: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
tts1-1  |   with autocast_func(enabled=False):
tts1-1  | Serving TTS on :5003
```

And it's only properly up and running once Serving TTS on :5003 is seen.

## Control Script

A convenience script `control.py` is provided to interact with the API from the command line. It requires the `requests` library (`pip install requests`).

### Setup
The script uses the `TTS_AUTHORIZATION_TOKEN` environment variable for authentication. It defaults to `vote_goof_2024` if not set.

### Commands

#### 1. Generate Audio
Interactively select a voice and generate an `.ogg` file.
```bash
python control.py generate
```
*   **Options:**
    *   `--pitch <int>`: Shift the pitch (default: 0).
    *   `--filters "filter1|filter2"`: Apply special filters (e.g., `silicon`).
    *   `--radio`: Automatically generates the base audio and then fetches the radio-processed version. Saves both as `output_<id>.ogg` and `output_radio_<id>.ogg`.

#### 2. Toggle Logging
Switch the logging level of the API and all backend services between `INFO` and `DEBUG` at runtime.
```bash
python control.py toggle-logging
```

### Manual Testing
To test manually via curl:
```bash
curl -X GET "http://localhost:5002/tts?identifier=job9001&voice=Example%20Woman&pitch=0" \
     -H "Authorization: vote_goof_2024" \
     -H "Content-Type: application/json"  \
     -d '{"text":"Yeah apc destroyed mission accomplished!"}'\
     --output test.ogg
```

Then play with your choice of player

You can view hte haproxy stats at localhost:8008

The docker compose file is a multi stage build of system deps, python deps and then the actual code

if you want to rebuild python deps for some reason you can with
`docker compose build --build-arg CACHEBUST=$(date +%s)`
