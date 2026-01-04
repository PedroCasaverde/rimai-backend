FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Instalar Python y dependencias del sistema (incluyendo build tools)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Actualizar pip
RUN pip3 install --upgrade pip setuptools wheel

# Instalar PyTorch con CUDA primero
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Instalar dependencias de audio/ML
RUN pip3 install --no-cache-dir \
    numpy \
    numba \
    librosa \
    soundfile

# Instalar Whisper y pyannote
RUN pip3 install --no-cache-dir \
    openai-whisper \
    pyannote.audio

# Instalar utilidades
RUN pip3 install --no-cache-dir \
    pydub \
    boto3 \
    ffmpeg-python

COPY main.py .

CMD ["python3", "main.py"]