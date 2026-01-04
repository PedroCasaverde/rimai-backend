FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Instalar Python y dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Crear symlink para python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Instalar PyTorch con CUDA
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar el resto de dependencias
RUN pip3 install --no-cache-dir \
    openai-whisper \
    pyannote.audio \
    pydub \
    boto3

COPY main.py .

CMD ["python", "main.py"]