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
    iproute2 \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# 2. Préparation du répertoire de travail
WORKDIR /aria

# 3. Installation et configuration de UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy
# Gros wheels (torch, cudnn...) : évite les timeouts réseau
ENV UV_HTTP_TIMEOUT=600
ENV UV_HTTP_RETRIES=5

# 4. Création de l'environnement virtuel INTERNE (Performance Max)
# On le place hors du répertoire /aria pour éviter la latence du montage Windows
RUN uv venv /venv_internal
ENV PATH="/venv_internal/bin:$PATH"

# 5. Installation des dépendances lourdes dans l'environnement interne
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN uv pip install onnxruntime-gpu==1.20.0 sherpa-onnx==1.12.28 nvidia-cudnn-cu12==9.1.0.70
RUN uv pip install wheel setuptools

ARG USE_REQ_LOCK=true
COPY requirements.txt .
COPY requirements.lock .
RUN if [ "$USE_REQ_LOCK" = "true" ] ; then \
        echo "Installation via requirements.lock" && \
        uv pip install -r requirements.lock; \
    else \
        echo "Installation via requirements.txt" && \
        uv pip install -r requirements.txt; \
    fi

# Paquets absents du requirements.lock (présents dans requirements.txt)
RUN uv pip install "anthropic>=0.40.0" faster-whisper==1.2.1

# Option : llama-cpp-python compilé avec CUDA (LLM local sur GPU). Désactivé par défaut : la version CPU du lock est conservée.
# Placé après les installations ci-dessus pour que le build par défaut garde son cache.
# Aucun GPU n'est requis pour compiler (nvcc est dans l'image), seulement pour exécuter.
# CUDA_ARCH : architecture ciblée, ex. 89 (RTX Ada) ou "75;86". Vide = toutes les architectures (compilation bien plus longue).
ARG LLAMA_CUDA=false
ARG CUDA_ARCH=""
RUN if [ "$LLAMA_CUDA" = "true" ] ; then \
        echo "llama-cpp-python : compilation avec CUDA (architectures : ${CUDA_ARCH:-toutes})" && \
        apt-get update && apt-get install -y cmake build-essential && rm -rf /var/lib/apt/lists/* && \
        ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
        export CMAKE_ARGS="-DGGML_CUDA=on${CUDA_ARCH:+ -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}}" \
               FORCE_CMAKE=1 CUDA_HOME=/usr/local/cuda LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
               LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} && \
        uv pip install --reinstall-package llama-cpp-python --no-cache --no-binary llama-cpp-python \
            "$(grep -i '^llama-cpp-python==' requirements.lock)"; \
    else \
        echo "llama-cpp-python : version CPU (LLAMA_CUDA=false)"; \
    fi

# Gestion optionnelle de Flash Attention (skip si échec de build)
ARG BUILD_FLASH_ATTN=true
RUN if [ "$BUILD_FLASH_ATTN" = "true" ] ; then \
        apt-get update && apt-get install -y ninja-build && \
        export MAX_JOBS=1 && \
        uv pip install flash-attn==2.7.4.post1 --no-build-isolation || echo "Flash-attn build failed, skipping..."; \
    fi

# 6. Verrouillage du Linker Linux (Cibles pointant vers /venv_internal)
RUN echo "/venv_internal/lib/python3.12/site-packages/onnxruntime/capi" > /etc/ld.so.conf.d/onnxruntime.conf && \
    echo "/venv_internal/lib/python3.12/site-packages/nvidia/cudnn/lib" >> /etc/ld.so.conf.d/onnxruntime.conf && \
    echo "/venv_internal/lib/python3.12/site-packages/nvidia/cublas/lib" >> /etc/ld.so.conf.d/onnxruntime.conf

# Lien symbolique spécifique pour sherpa-onnx
RUN ln -sf /venv_internal/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so \
           /venv_internal/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so

RUN ldconfig

# 7. Configuration de l'environnement d'exécution
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
# Chemins mis à jour pour le venv interne
ENV LD_LIBRARY_PATH="/venv_internal/lib/python3.12/site-packages/nvidia/cudnn/lib:/venv_internal/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

ENV UV_PROJECT_ENVIRONMENT=/venv_internal

# Raccourci bashrc
RUN echo 'source /venv_internal/bin/activate' >> ~/.bashrc

# 8. Point d'entrée
ENTRYPOINT ["python", "app.py"]