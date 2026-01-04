FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN apt update && apt upgrade -y \
    && apt install -y git opus-tools python3-pip python3.12-venv \
    portaudio19-dev python3-tk pulseaudio pulseaudio-utils vim ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /aria

RUN git clone -b WebHMI https://github.com/brag00n/aria.git .

# Création de l'environnement virtuel
RUN python3 -m venv /venv
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"

# Installation des dépendances
COPY requirements.txt .
RUN grep -Ev 'PyAudio==0.2.14|onnxruntime==1.20.1' requirements.txt > requirements_filtered.txt \
    && pip install --no-cache-dir -r requirements_filtered.txt

RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1 
RUN pip install --no-cache-dir onnxruntime==1.20.1 

# Installation des bibliothèques NVIDIA et des moteurs de calcul
RUN pip install --no-cache-dir PyAudio==0.2.14 \
    && pip install --no-cache-dir gradio==4.44.1 pydub pydantic==2.10.6 \
    && pip install --no-cache-dir faster-whisper==1.2.1 \
    && pip install --no-cache-dir nvidia-cublas-cu12 nvidia-cudnn-cu12 \
    && pip install --no-cache-dir "huggingface-hub<1.0"

# --- AJOUT DES LIENS SYMBOLIQUES POUR CUDNN 9 (ARCHITECTURE ADA) ---
RUN cd /venv/lib/python3.12/site-packages/nvidia/cudnn/lib/ && \
    ln -sf libcudnn_cnn_infer.so.9 libcudnn_cnn.so.9 && \
    ln -sf libcudnn_ops_infer.so.9 libcudnn_ops.so.9 && \
    ln -sf libcudnn_graph_infer.so.9 libcudnn_graph.so.9 && \
    ln -sf libcudnn_engines_runtime_helper.so.9 libcudnn_engines_runtime_helper.so.9

# --- CONFIGURATION DES VARIABLES D'ENVIRONNEMENT GPU ---
ENV LD_LIBRARY_PATH="/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/lib/python3.12/site-packages/nvidia/cudart/lib:/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

EXPOSE 7860

# Lancement de l'application
ENTRYPOINT ["bash", "-c", "source /venv/bin/activate && exec python app.py"]