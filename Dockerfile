# Dockerfile - CosyVoice2 RunPod Serverless
# Base: PyTorch 2.5 + CUDA 12.4
FROM runpod/pytorch:2.5.1-py3.10-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y \
    sox libsox-dev git git-lfs ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone CosyVoice repo (with submodules)
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /workspace/CosyVoice \
    && cd /workspace/CosyVoice \
    && git submodule update --init --recursive

# Install CosyVoice dependencies
RUN cd /workspace/CosyVoice \
    && pip install --no-cache-dir -r requirements.txt

# Install RunPod + extra deps
COPY requirements.txt /workspace/app/requirements.txt
RUN pip install --no-cache-dir -r /workspace/app/requirements.txt

# Download model (cached in Docker layer)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='/workspace/pretrained_models/CosyVoice2-0.5B')"

# Optional: download ttsfrd for better text normalization
# RUN cd /workspace/CosyVoice/pretrained_models && \
#     python3 -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='CosyVoice-ttsfrd')" && \
#     cd CosyVoice-ttsfrd && unzip -o resource.zip -d . && \
#     pip install ttsfrd_dependency-0.1-py3-none-any.whl ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl

# Copy handler
COPY rp_handler.py /workspace/app/rp_handler.py
COPY preprocess.py /workspace/app/preprocess.py

# Set env
ENV COSYVOICE_ROOT=/workspace/CosyVoice
ENV MODEL_DIR=/workspace/pretrained_models/CosyVoice2-0.5B
ENV PYTHONPATH=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS

CMD ["python3", "/workspace/app/rp_handler.py"]
