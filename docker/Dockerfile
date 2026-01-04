FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Evitar prompts interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# 1. Instalar herramientas del sistema
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

# 2. Actualizar pip (vital para resolver dependencias modernas)
RUN pip3 install --upgrade pip setuptools wheel

# 3. Instalar PyTorch (Versión compatible con CUDA 11.8)
# Lo instalamos primero porque es el más pesado
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Instalar Resto de Librerías (CON COMILLAS en las versiones < )
# Agrupamos todo en un solo comando para que pip resuelva conflictos de una vez
# Nota el uso de "numpy<2.0.0" para evitar el error de AttributeError
RUN pip3 install --no-cache-dir \
    "numpy<2.0.0" \
    "pandas<2.2.0" \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    audioread==3.0.1 \
    scipy==1.11.4 \
    pyannote.audio==3.1.1 \
    openai-whisper==20231117 \
    pydub==0.25.1 \
    boto3==1.34.34 \
    ffmpeg-python==0.2.0

# Copiar código
COPY main.py .

# Comando de ejecución
CMD ["python3", "main.py"]