FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN apt update && apt upgrade -y \
    && apt install -y git opus-tools \
    python3-pip python3.12-venv \
    && git clone https://github.com/brag00n/aria.git

WORKDIR /aria

RUN python3 -m venv /venv
ENV VIRTUAL_ENV=venv
ENV PATH=/venv/bin:$PATH

RUN bash -c "source /venv/bin/activate && \
 pip install --no-cache-dir -r <(grep -Ev 'PyAudio==0.2.14|onnxruntime==1.20.1' requirements.txt) && \
 pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1 && \
 pip install --no-cache-dir onnxruntime==1.20.1 && \
 apt install -y portaudio19-dev && \
 apt install -y python3-tk && \
 apt install -y pulseaudio pulseaudio-utils && \
 pip install -t --no-cache-dir -r <(cat requirements_client.txt) && \
 pip install --no-cache-dir PyAudio==0.2.14"

RUN apt install -y vim

RUN echo 'source /venv/bin/activate' >> ~/.bashrc

ENTRYPOINT ["bash", "-c", "source /venv/bin/activate && exec python app.py"]