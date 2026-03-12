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

# Install Python dependencies
RUN pip install --no-cache-dir flask waitress tqdm pysbd blake3 stftpitchshift \
    && pip install --no-cache-dir git+https://github.com/ysharma3501/LavaSR.git \
    && pip install --no-cache-dir faster-qwen3-tts


# Stage 3: Copy application code
FROM python-deps AS final

WORKDIR /workspace
# Copy application code last
COPY *.py ./
COPY *.wav ./
COPY blips_sfx/ /workspace/blips_sfx

# Set environment again if needed
ENV TORCH_CUDA_ARCH_LIST=8.9

# Optional default command
# CMD ["python", "your_main.py"]