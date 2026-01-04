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

# Actualizar pip
RUN pip3 install --upgrade pip setuptools wheel

# --- BLOQUE CRÍTICO: NUMPY ---
# Forzamos NumPy menor a 2.0 antes que nada.
# Esto evita que otras librerías intenten actualizarlo después.
RUN pip3 install --no-cache-dir "numpy<2.0.0"

# 1. Instalar PyTorch con CUDA (Compatible con NumPy 1.x)
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 2. Instalar dependencias de audio
RUN pip3 install --no-cache-dir \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    audioread==3.0.1 \
    scipy==1.11.4

# 3. Instalar pyannote (Diarización)
# Instalamos pandas explícitamente para evitar conflictos
RUN pip3 install --no-cache-dir pandas<2.2.0 
RUN pip3 install --no-cache-dir pyannote.audio==3.1.1

# 4. Instalar Whisper
RUN pip3 install --no-cache-dir openai-whisper==20231117

# 5. Instalar utilidades restantes
RUN pip3 install --no-cache-dir \
    pydub==0.25.1 \
    boto3==1.34.34 \
    ffmpeg-python==0.2.0

# 6. VERIFICACIÓN FINAL DE SEGURIDAD
# Ejecutamos esto al final para asegurar que nada actualizó numpy a la versión 2
RUN pip3 install --no-cache-dir "numpy<2.0.0" --force-reinstall

# Copiar código
COPY main.py .

# Comando de ejecución
CMD ["python3", "main.py"]