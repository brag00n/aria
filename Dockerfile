FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# 1. Installation des dépendances système (Audio + Python)
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    portaudio19-dev \
    libopus-dev \
    ffmpeg \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 2. Préparation du répertoire de travail
WORKDIR /aria
COPY . .

# 3. Installation de Python & Pip
RUN pip3 install --upgrade pip setuptools wheel

# 4. Installation des dépendances (Ordre stratégique pour le cache)
# On installe d'abord les gros morceaux (Torch)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Installation du reste via ton requirements unifié
RUN pip3 install --no-cache-dir -r requirements.txt

# Installation spécifique pour les perfs GPU (Flash Attn peut être long)
RUN pip3 install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# 5. Configuration Environnement
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

RUN apt install -y vim

RUN echo 'source /venv/bin/activate' >> ~/.bashrc

ENTRYPOINT ["bash", "-c", "source /venv/bin/activate && exec python app.py"]