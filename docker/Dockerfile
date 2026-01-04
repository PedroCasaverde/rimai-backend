# Usamos una imagen ligera de Python
FROM python:3.9-slim

# Instalar FFmpeg (necesario para audio) y git
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY main.py .

# Comando por defecto (se sobreescribe al lanzar el Job, pero dejamos este de base)
CMD ["python", "main.py"]