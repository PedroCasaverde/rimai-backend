FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Evitar que Linux haga preguntas durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# 1. Instalar herramientas del sistema necesarias para audio y compilación
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

# 2. Actualizar pip (vital para evitar errores de compilación)
RUN pip3 install --upgrade pip setuptools wheel

# 3. Instalar PyTorch COMPATIBLE con CUDA 11.8
# Lo instalamos en un paso separado porque es muy pesado
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Instalar todas las librerías de IA y Audio
# ¡IMPORTANTE!: Usamos comillas " " en las versiones con < para evitar error de sintaxis
RUN pip3 install --no-cache-dir \
    "numpy<2.0.0" \
    "pandas<2.2.0" \
    "huggingface_hub<0.22.0" \  <-- ¡ESTA ES LA LÍNEA NUEVA QUE NECESITAS!
    librosa==0.10.1 \
    soundfile==0.12.1 \
    audioread==3.0.1 \
    scipy==1.11.4 \
    pyannote.audio==3.1.1 \
    openai-whisper==20231117 \
    pydub==0.25.1 \
    boto3==1.34.34 \
    ffmpeg-python==0.2.0

# Copiar el código fuente
COPY main.py .

# Comando de arranque
CMD ["python3", "main.py"]