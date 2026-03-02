FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Evitar que Linux haga preguntas durante la instalación
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# 1. Instalar herramientas del sistema necesarias para audio y compilación
#    - ffmpeg: requerido por pydub para convertir audio
#    - libsndfile: requerido por librosa/soundfile
#    - QUITADOS: sox, libsox-dev, libsox-fmt-all (no se usan), git (no se usa en runtime)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Actualizar pip (vital para evitar errores de compilación)
RUN pip3 install --upgrade pip setuptools wheel

# 3. Instalar PyTorch COMPATIBLE con CUDA 12.4
#    - QUITADO: torchvision (no se usa en el proyecto)
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 4. Instalar todas las librerías de IA y Audio
#    CAMBIOS:
#    - openai-whisper → faster-whisper (4x más rápido, misma calidad)
#    - pyannote.audio actualizado a 3.3.2 (compatible con PyTorch 2.5)
#    - QUITADOS: ffmpeg-python (no se usa), audioread (librosa lo trae),
#      huggingface_hub pin (pyannote lo maneja)
RUN pip3 install --no-cache-dir \
    "numpy<2.0.0" \
    "pandas<2.2.0" \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    scipy==1.11.4 \
    pyannote.audio==3.3.2 \
    faster-whisper==1.1.0 \
    pydub==0.25.1 \
    boto3==1.34.34

# Copiar el código fuente
COPY main.py .

# Comando de arranque
CMD ["python3", "main.py"]