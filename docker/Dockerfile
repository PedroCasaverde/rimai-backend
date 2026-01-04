FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    git \
    libsndfile1 \
    libsndfile1-dev \
    sox \
    libsox-dev \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Actualizar pip y herramientas
RUN pip3 install --upgrade pip setuptools wheel

# 1. Instalar numpy primero (requerido por muchos paquetes)
RUN pip3 install --no-cache-dir numpy==1.24.3

# 2. Instalar PyTorch con CUDA
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. Instalar dependencias de audio (antes de pyannote)
RUN pip3 install --no-cache-dir \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    audioread==3.0.1 \
    scipy==1.11.4

# 4. Instalar dependencias de pyannote una por una
RUN pip3 install --no-cache-dir \
    pyannote.core==5.0.0 \
    pyannote.database==5.0.1 \
    pyannote.metrics==3.2.1

# 5. Instalar pyannote.audio (ahora debería funcionar)
RUN pip3 install --no-cache-dir pyannote.audio==3.1.1

# 6. Instalar Whisper
RUN pip3 install --no-cache-dir openai-whisper==20231117

# 7. Instalar utilidades restantes
RUN pip3 install --no-cache-dir \
    pydub==0.25.1 \
    boto3==1.34.34 \
    ffmpeg-python==0.2.0

# Copiar código
COPY main.py .

# Comando de ejecución
CMD ["python3", "main.py"]