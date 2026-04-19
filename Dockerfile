# Stage 1: Install system dependencies
FROM docker.io/pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel AS system-deps

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
    sox \
    && rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST=8.9


# Stage 2: Install Python dependencies
FROM system-deps AS python-deps
ARG CACHEBUST=1 # Allows us to bust docker build cache.
WORKDIR /workspace
ENV TORCH_CUDA_ARCH_LIST=8.9
RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu126torch2.7-cp311-cp311-linux_x86_64.whl \
RUN pip install --no-cache-dir nano-vllm-voxcpm
# Install Python dependencies
RUN pip install --no-cache-dir flask waitress tqdm pysbd blake3 stftpitchshift pydub fastapi uvicorn pandas 



# Stage 3: Copy application code
FROM python-deps AS final

WORKDIR /workspace
# Copy application code last
COPY *.py ./
COPY *.js /workspace/static/
COPY *.wav ./
COPY *.ogg ./
COPY blips_sfx/ /workspace/blips_sfx

# Set environment again if needed
ENV TORCH_CUDA_ARCH_LIST=8.9

# Optional default command
# CMD ["python", "your_main.py"]
