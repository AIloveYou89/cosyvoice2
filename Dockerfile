# Dockerfile - CosyVoice2 RunPod Serverless
# Base: PyTorch 2.6 + CUDA 12.6 (same as SparkTTS)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

WORKDIR /workspace

# System deps (sox for CosyVoice, g++ for pyworld C extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    sox libsox-dev git git-lfs ffmpeg libsndfile1 \
    g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

# Torchaudio match torch 2.6
RUN pip install --upgrade pip && pip install torchaudio==2.6.0

# Clone CosyVoice repo (with submodules)
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /workspace/CosyVoice \
    && cd /workspace/CosyVoice \
    && git submodule update --init --recursive

# Install CosyVoice dependencies
# Strip packages not needed for TTS inference (cause import crashes on runtime image)
RUN cd /workspace/CosyVoice \
    && sed -i '/openai-whisper/d' requirements.txt \
    && sed -i '/deepspeed/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# Install RunPod + extra deps
COPY requirements.txt /workspace/app/requirements.txt
RUN pip install --no-cache-dir -r /workspace/app/requirements.txt

# Fix: remove packages that crash on runtime image (no CUDA_HOME/nvcc)
# deepspeed needs CUDA_HOME, torchvision incompatible with torch 2.6
# Neither is needed for CosyVoice TTS inference
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 \
    && pip uninstall -y torchvision deepspeed 2>/dev/null; true

# Smoke test: verify CosyVoice imports work
RUN python3 -c "from cosyvoice.cli.cosyvoice import AutoModel; print('Import OK')"

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
