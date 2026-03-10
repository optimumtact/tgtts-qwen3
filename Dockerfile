FROM docker.io/pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

WORKDIR /workspace

RUN apt update && \
  apt install -y \
  libsox-dev \
  ffmpeg \
  build-essential \
  cmake \
  libasound-dev \
  portaudio19-dev \
  libportaudio2 \
  libportaudiocpp0 \
  nvidia-cuda-toolkit \
  libvorbis-dev \
  git \
  sox
COPY *.py ./
COPY *.wav ./
COPY blips_sfx/ /workspace/blips_sfx
ENV TORCH_CUDA_ARCH_LIST=8.9

RUN pip install flask waitress tqdm pysbd blake3 stftpitchshift
RUN pip install git+https://github.com/ysharma3501/LavaSR.git
RUN pip install faster-qwen3-tts