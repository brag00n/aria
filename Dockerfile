# Utilisation de l'image de base NVIDIA avec CUDA 12.8 et cuDNN [cite: 1]
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
    iproute2 \
    iputils-ping \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Préparation du répertoire de travail [cite: 2]
WORKDIR /aria

# 3. Installation et configuration de UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

# 4. Création de l'environnement virtuel INTERNE (Performance Max)
RUN uv venv /venv_internal
ENV PATH="/venv_internal/bin:$PATH"

# 5. Installation des dépendances lourdes
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN uv pip install onnxruntime-gpu==1.20.1 sherpa-onnx==1.12.23 nvidia-cudnn-cu12==9.1.0.70
RUN uv pip install wheel setuptools

# --- NOUVEAU : COMPILATION LLAMA-CPP-PYTHON OPTIMISÉE ---
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CMAKE=1
ENV CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on -DGGML_FLASH_ATTN=on"

RUN uv pip install -U pip && \
    uv pip install llama-cpp-python==0.3.16 --no-binary llama-cpp-python --no-cache-dir

# Installation des autres dépendances [cite: 3, 4, 5, 6]
ARG USE_REQ_LOCK=true
COPY requirements.txt .
COPY requirements.lock .
RUN if [ "$USE_REQ_LOCK" = "true" ] ; \
    then \
        uv pip install -r requirements.lock; \
    else \
        uv pip install -r requirements.txt; \
    fi

# Gestion de Flash Attention (pour les autres modèles) [cite: 7, 8]
ARG BUILD_FLASH_ATTN=true
RUN if [ "$BUILD_FLASH_ATTN" = "true" ] ; \
    then \
        apt-get update && apt-get install -y ninja-build && \
        export MAX_JOBS=1 && \
        uv pip install flash-attn==2.7.4.post1 --no-build-isolation || \
        echo "Flash-attn build failed, skipping..."; \
    fi

# 6. Verrouillage du Linker Linux
RUN ldconfig

# 7. Configuration de l'environnement d'exécution
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH="/venv_internal/lib/python3.12/site-packages/nvidia/cudnn/lib:/venv_internal/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"
ENV UV_PROJECT_ENVIRONMENT=/venv_internal

# 8. Point d'entrée
ENTRYPOINT ["python", "app.py"]