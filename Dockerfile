# Utilisation de l'image de base NVIDIA avec CUDA 12.8 et cuDNN
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# 1. Installation des dépendances système (Audio, Python et Compilateurs)
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    portaudio19-dev \
    libopus-dev \
    ffmpeg \
    vim \
    libgomp1 \
    libatlas-base-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Préparation du répertoire de travail
WORKDIR /aria
COPY . .

# 3. Installation et configuration de UV pour une gestion propre du venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# On force le mode copie pour éviter les problèmes de liens entre systèmes de fichiers
ENV UV_LINK_MODE=copy

# 4. Création de l'environnement virtuel (venv)
RUN uv venv /aria/.venv
ENV PATH="/aria/.venv/bin:$PATH"

# 5. Installation stratégique des dépendances lourdes (Figeage des versions)
# On installe d'abord Torch avec l'index CUDA 12.4 (stable pour cette config)
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# On installe les binaires ONNX (TTS sherpa) et cuDNN (STT) qui ont résolu l'erreur de symboles
RUN uv pip install onnxruntime-gpu==1.20.1 sherpa-onnx==1.12.23 nvidia-cudnn-cu12==9.1.0.70

# On installe le reste via ton requirements.txt (UV ignorera ce qui est déjà installé)
RUN uv pip install -r requirements.txt

# Installation de Flash Attention (nécessite les outils de build devel déjà présents)
RUN uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# 6. Verrouillage du Linker Linux (Correction définitive des .so)
# On crée les accès pour que le système trouve les libs dans le venv
RUN echo "/aria/.venv/lib/python3.12/site-packages/onnxruntime/capi" > /etc/ld.so.conf.d/onnxruntime.conf && \
    echo "/aria/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib" >> /etc/ld.so.conf.d/onnxruntime.conf && \
    echo "/aria/.venv/lib/python3.12/site-packages/nvidia/cublas/lib" >> /etc/ld.so.conf.d/onnxruntime.conf

# Lien symbolique interne spécifique requis par sherpa-onnx
RUN ln -sf /aria/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so \
           /aria/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so

# Mise à jour du cache des bibliothèques système
RUN ldconfig

# 7. Configuration de l'environnement d'exécution
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Priorité aux bibliothèques CUDA du venv pour éviter le CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
ENV LD_LIBRARY_PATH="/aria/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/aria/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

# Ajout du raccourci dans le bashrc pour les sessions interactives
RUN echo 'source /aria/.venv/bin/activate' >> ~/.bashrc

# Pur les dignostics réseau (optionnel)
RUN apt-get update && apt-get install -y iproute2 iputils-ping

# 8. Point d'entrée
ENTRYPOINT ["python", "app.py"]