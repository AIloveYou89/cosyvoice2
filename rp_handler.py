# rp_handler.py - CosyVoice2 RunPod Serverless Handler
# Adapted from Vi-SparkTTS rp_handler pattern
# Model: FunAudioLLM/CosyVoice2-0.5B (Alibaba)
# Features: Zero-shot voice clone, streaming 150ms, 9 languages
# NOTE: Vietnamese chưa chính thức hỗ trợ — cần test thực tế

import os, io, uuid, base64, numpy as np, soundfile as sf, torch, torchaudio
import runpod
import gc
import logging
import re
import sys
import time
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = os.getenv("MODEL_DIR", "pretrained_models/CosyVoice2-0.5B")
HF_TOKEN = os.getenv("HF_TOKEN", "")

if torch.cuda.is_available():
    DEVICE = "cuda"
    logger.info(f"[INIT] CUDA available: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    logger.error("[INIT] CUDA/GPU NOT available — will be very slow!")
TARGET_SR = 22050  # CosyVoice2 default sample rate

# ============================================================
# PREPROCESS MODULE — Vietnamese text normalization
# Giữ nguyên logic từ SparkTTS, thêm CosyVoice-specific tweaks
# ============================================================
try:
    from num2words import num2words
    HAS_NUM2WORDS = True
except ImportError:
    HAS_NUM2WORDS = False
    logger.warning("[PREPROCESS] num2words not installed, numbers won't be converted")

MIN_CHARS_PER_CHUNK = 60
MAX_CHARS_PER_CHUNK = 180  # CosyVoice handles longer chunks better
OPTIMAL_CHUNK_SIZE = 120
PUNCS = r".?!…"

_number_pattern = re.compile(r"(\d{1,3}(?:\.\d{3})*)(?:\s*(%|[^\W\d_]+))?", re.UNICODE)
_whitespace_pattern = re.compile(r"\s+")
_comma_pattern = re.compile(r"\s*,\s*")
_punct_spacing_pattern = re.compile(r"\s+([,;:])")
_repeated_punct_pattern = re.compile(rf"[{PUNCS}]{{2,}}")
_punct_no_space_pattern = re.compile(rf"([{PUNCS}])(?=\S)")

def normalize_text_vn(text: str) -> str:
    text = text.strip()
    text = _whitespace_pattern.sub(" ", text)
    text = _comma_pattern.sub(", ", text)
    # NOTE: Không lowercase — CosyVoice2 xử lý mixed case tốt hơn

    if HAS_NUM2WORDS:
        def repl_number_with_unit(m):
            num_str = m.group(1).replace(".", "")
            unit = m.group(2) or ""
            try:
                return num2words(int(num_str), lang="vi") + (" " + unit if unit else "")
            except:
                return m.group(0)
        text = _number_pattern.sub(repl_number_with_unit, text)

    text = _punct_spacing_pattern.sub(r"\1", text)
    text = _repeated_punct_pattern.sub(lambda m: m.group(0)[0], text)
    text = _punct_no_space_pattern.sub(r"\1 ", text)
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    parts = re.split(rf"(?<=[{PUNCS}])\s+", text)
    return [p.strip() for p in parts if p.strip() and not re.fullmatch(rf"[{PUNCS}]+", p)]

def ensure_punctuation(s: str) -> str:
    s = s.strip()
    if not s.endswith(tuple(PUNCS)):
        s += "."
    return s

def ensure_leading_dot(s: str) -> str:
    """Thủ thuật dấu chấm đầu chunk — giúp model đọc chuẩn hơn."""
    s = s.lstrip()
    if s and s[0] not in PUNCS:
        return ". " + s
    return s

def smart_chunk_split(text: str) -> List[str]:
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        word_len = len(word) + 1
        if current_length + word_len > MAX_CHARS_PER_CHUNK and current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) < MIN_CHARS_PER_CHUNK and chunks:
                prev = chunks[-1]
                if len(prev) + len(chunk_text) + 1 <= MAX_CHARS_PER_CHUNK:
                    chunks[-1] = prev + " " + chunk_text
                    current_chunk = [word]
                    current_length = word_len
                    continue
            chunks.append(chunk_text)
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) < MIN_CHARS_PER_CHUNK and chunks:
            chunks[-1] += " " + chunk_text
        else:
            chunks.append(chunk_text)

    return chunks

def preprocess_text(text: str) -> List[str]:
    clean = normalize_text_vn(text)
    sentences = split_into_sentences(clean)

    if not sentences:
        s = ensure_punctuation(clean)
        return [ensure_leading_dot(s)]

    chunks = []
    buffer = ""

    for i, sent in enumerate(sentences, 1):
        sent = ensure_punctuation(sent)
        sent = re.sub(r'^([^\w]*\w[^,]{0,10}),\s*', r'\1 ', sent)

        if i <= 5:
            if len(sent) > MAX_CHARS_PER_CHUNK:
                chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(sent)])
            else:
                chunks.append(ensure_leading_dot(sent))
        else:
            if len(sent) > MAX_CHARS_PER_CHUNK:
                if buffer:
                    chunks.append(ensure_leading_dot(buffer))
                    buffer = ""
                chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(sent)])
            else:
                if buffer and len(buffer) + len(sent) + 1 > OPTIMAL_CHUNK_SIZE:
                    chunks.append(ensure_leading_dot(buffer))
                    buffer = sent
                elif buffer:
                    buffer += " " + sent
                else:
                    buffer = sent

    if buffer:
        if len(buffer) > MAX_CHARS_PER_CHUNK:
            chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(buffer)])
        else:
            chunks.append(ensure_leading_dot(ensure_punctuation(buffer)))

    return chunks

# ============================================================
# AUDIO HELPERS (từ SparkTTS pattern)
# ============================================================
def join_with_pause(a: np.ndarray, b: np.ndarray, sr: int,
                    gap_sec: float = 0.15, fade_sec: float = 0.08):
    gap_n = max(int(sr * 0.05), int(sr * gap_sec))
    fade_n = max(0, int(sr * fade_sec))

    if a.ndim == 2:
        silence = np.zeros((gap_n, a.shape[1]), dtype=np.float32)
    else:
        silence = np.zeros(gap_n, dtype=np.float32)

    if fade_n <= 0 or len(a) < fade_n or len(b) < fade_n:
        return np.concatenate([a, silence, b], axis=0)

    fade_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    fade_in = fade_out[::-1]
    if a.ndim == 2:
        fade_out = fade_out[:, None]
        fade_in = fade_in[:, None]

    a_tail = a[-fade_n:] * fade_out
    b_head = b[:fade_n] * fade_in
    return np.concatenate([a[:-fade_n], a_tail, silence, b_head, b[fade_n:]], axis=0)

def normalize_peak(x: np.ndarray, peak=0.95):
    if x is None or x.size == 0:
        return x
    max_val = float(np.max(np.abs(x)))
    if max_val > 0.98:
        return (x * (peak / max_val)).astype(np.float32)
    return x

def resample_np(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    wav = torch.from_numpy(x).unsqueeze(0)
    out = torchaudio.functional.resample(wav, sr_in, sr_out)
    return out.squeeze(0).numpy()

# ============================================================
# NETWORK VOLUME — prompt audio detection
# ============================================================
def detect_network_volume_path() -> Optional[str]:
    possible_roots = [
        "/runpod-volume",
        os.getenv("RUNPOD_VOLUME_PATH"),
        os.getenv("NV_ROOT"),
    ]
    for root in possible_roots:
        if root and os.path.exists(root) and os.path.isdir(root):
            logger.info(f"[VOLUME] Found Network Volume: {root}")
            return root
    logger.warning("[VOLUME] No Network Volume detected")
    return None

NV_ROOT = detect_network_volume_path()

def find_prompt_audio() -> Optional[str]:
    if not NV_ROOT:
        return None

    possible_paths = [
        os.path.join(NV_ROOT, "workspace/consent_audio.wav"),
        os.path.join(NV_ROOT, "consent_audio.wav"),
        os.path.join(NV_ROOT, "audio/consent_audio.wav"),
        os.path.join(NV_ROOT, "prompts/consent_audio.wav"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                audio_data, sr = sf.read(path)
                duration = len(audio_data) / sr
                if len(audio_data) == 0 or duration < 1.0:
                    continue
                max_amp = np.max(np.abs(audio_data))
                if max_amp < 0.001:
                    continue
                logger.info(f"[PROMPT] Found: {path} ({duration:.2f}s, sr={sr})")
                return path
            except Exception as e:
                logger.error(f"[PROMPT] Error reading {path}: {e}")

    logger.warning("[PROMPT] No valid prompt audio found")
    return None

OUT_DIR = os.path.join(NV_ROOT, "jobs") if NV_ROOT else "/tmp/jobs"

# ============================================================
# MODEL LOADING — CosyVoice2
# ============================================================
cosyvoice = None
PROMPT_PATH = None
DEFAULT_PROMPT_TEXT = os.getenv(
    "PROMPT_TEXT",
    "Tôi là chủ sở hữu giọng nói này."
)

try:
    logger.info(f"[INIT] Loading CosyVoice2 from {MODEL_DIR}")
    init_start = time.time()

    # Download model if not exists
    model_path = MODEL_DIR
    if not os.path.exists(model_path):
        # Check Network Volume first (only if MODEL_DIR is relative)
        nv_model = os.path.join(NV_ROOT, os.path.basename(MODEL_DIR)) if NV_ROOT else None
        if nv_model and os.path.exists(nv_model):
            model_path = nv_model
            logger.info(f"[INIT] Using model from Network Volume: {model_path}")
        else:
            logger.info("[INIT] Downloading model from HuggingFace...")
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(
                'FunAudioLLM/CosyVoice2-0.5B',
                local_dir=MODEL_DIR,
                token=HF_TOKEN if HF_TOKEN else None
            )
            logger.info(f"[INIT] Model downloaded to: {model_path}")

    # List model dir contents for debugging
    if os.path.exists(model_path):
        logger.info(f"[INIT] Model dir contents: {os.listdir(model_path)}")
    else:
        logger.error(f"[INIT] Model path does NOT exist: {model_path}")

    # Add CosyVoice to path
    COSYVOICE_ROOT = os.getenv("COSYVOICE_ROOT", "/workspace/CosyVoice")
    if os.path.exists(COSYVOICE_ROOT):
        sys.path.insert(0, COSYVOICE_ROOT)
        matcha_path = os.path.join(COSYVOICE_ROOT, "third_party/Matcha-TTS")
        if os.path.exists(matcha_path):
            sys.path.insert(0, matcha_path)
        logger.info(f"[INIT] CosyVoice code path: {COSYVOICE_ROOT}")
        logger.info(f"[INIT] CosyVoice dir contents: {os.listdir(COSYVOICE_ROOT)}")
    else:
        logger.error(f"[INIT] CosyVoice code not found at {COSYVOICE_ROOT}")
        raise RuntimeError(f"CosyVoice code not found at {COSYVOICE_ROOT}")

    logger.info("[INIT] Importing CosyVoice AutoModel...")
    from cosyvoice.cli.cosyvoice import AutoModel as CosyAutoModel
    logger.info("[INIT] Import successful, initializing model...")

    cosyvoice = CosyAutoModel(model_dir=model_path, load_jit=False, load_trt=False)
    logger.info(f"[INIT] Model type: {type(cosyvoice)}")

    PROMPT_PATH = find_prompt_audio()
    if PROMPT_PATH:
        logger.info(f"[INIT] Prompt audio ready: {PROMPT_PATH}")
    else:
        logger.warning("[INIT] No prompt audio — will use default voice")

    init_time = time.time() - init_start
    logger.info(f"[INIT] Model loaded in {init_time:.2f}s")

    # Warm-up (skip if no prompt audio — avoid crash on missing default wav)
    if PROMPT_PATH:
        logger.info("[INIT] Warming up with prompt audio...")
        try:
            for i, j in enumerate(cosyvoice.inference_zero_shot(
                ". xin chào.",
                DEFAULT_PROMPT_TEXT,
                PROMPT_PATH,
                stream=False
            )):
                pass
            torch.cuda.empty_cache()
            logger.info("[INIT] Warmup complete")
        except Exception as e:
            logger.warning(f"[INIT] Warmup failed (non-critical): {e}")
    else:
        logger.info("[INIT] Skipping warmup (no prompt audio)")

except Exception as e:
    logger.error(f"[INIT] FATAL: Model loading failed: {e}", exc_info=True)
    import traceback
    traceback.print_exc()
    # Don't exit — let RunPod start so we can see error in logs
    # Handler will return error if cosyvoice is None

# ============================================================
# HANDLER
# ============================================================
def handler(job):
    job_start = time.time()

    if cosyvoice is None:
        return {"error": "Model failed to load. Check container logs for details."}

    inp = job.get("input", {}) or {}
    text = (inp.get("text") or "").strip()

    if not text:
        return {"error": "Missing 'text'."}

    if len(text) > 500000:
        return {"error": "Text too long (max 500,000 chars)"}

    # Parameters
    gap_sec = max(0.05, min(1.0, float(inp.get("gap_sec", 0.15))))
    fade_sec = max(0.02, min(0.5, float(inp.get("fade_sec", 0.08))))
    ret_mode = inp.get("return", "path")
    outfile = inp.get("outfile")
    output_sr = int(inp.get("output_sr", TARGET_SR))

    # Voice clone config
    custom_prompt_path = inp.get("prompt_path")
    prompt_text = inp.get("prompt_text", DEFAULT_PROMPT_TEXT)

    # Inference mode: zero_shot (voice clone) or cross_lingual
    inference_mode = inp.get("mode", "zero_shot")

    # Instruction (for instruct2 mode: dialect, emotion, speed)
    instruction = inp.get("instruction")

    active_prompt_path = None
    if custom_prompt_path and os.path.exists(custom_prompt_path):
        active_prompt_path = custom_prompt_path
    elif PROMPT_PATH:
        active_prompt_path = PROMPT_PATH

    if not active_prompt_path:
        logger.warning("[HANDLER] No prompt audio — using cross_lingual mode fallback")
        inference_mode = "cross_lingual"

    # Preprocess
    preprocess_start = time.time()
    try:
        chunks = preprocess_text(text)
        preprocess_time = time.time() - preprocess_start
        logger.info(f"[HANDLER] {len(chunks)} chunks ({preprocess_time:.2f}s)")
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}

    full_audio = None
    sr = None
    successful_chunks = 0

    generation_start = time.time()

    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        chunk_start = time.time()
        logger.info(f"[HANDLER] Chunk {idx+1}/{len(chunks)}: {len(chunk)} chars")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Generate audio based on mode
            chunk_audio = None

            if inference_mode == "zero_shot" and active_prompt_path:
                for i, j in enumerate(cosyvoice.inference_zero_shot(
                    chunk,
                    prompt_text,
                    active_prompt_path,
                    stream=False
                )):
                    audio_tensor = j['tts_speech']
                    chunk_sr = cosyvoice.sample_rate
                    chunk_audio = audio_tensor.squeeze().cpu().numpy()
                    if sr is None:
                        sr = chunk_sr

            elif inference_mode == "instruct2" and instruction and active_prompt_path:
                for i, j in enumerate(cosyvoice.inference_instruct2(
                    chunk,
                    instruction,
                    active_prompt_path,
                    stream=False
                )):
                    audio_tensor = j['tts_speech']
                    chunk_sr = cosyvoice.sample_rate
                    chunk_audio = audio_tensor.squeeze().cpu().numpy()
                    if sr is None:
                        sr = chunk_sr

            else:  # cross_lingual fallback
                for i, j in enumerate(cosyvoice.inference_cross_lingual(
                    chunk,
                    active_prompt_path or os.path.join(os.getenv("COSYVOICE_ROOT", "/workspace/CosyVoice"), 'asset/zero_shot_prompt.wav'),
                    stream=False
                )):
                    audio_tensor = j['tts_speech']
                    chunk_sr = cosyvoice.sample_rate
                    chunk_audio = audio_tensor.squeeze().cpu().numpy()
                    if sr is None:
                        sr = chunk_sr

            if chunk_audio is not None and len(chunk_audio) > 0:
                # Resample if needed
                if sr != output_sr:
                    chunk_audio = resample_np(chunk_audio, sr, output_sr)

                if full_audio is None:
                    full_audio = chunk_audio
                else:
                    full_audio = join_with_pause(full_audio, chunk_audio, output_sr, gap_sec, fade_sec)

                successful_chunks += 1
                chunk_time = time.time() - chunk_start
                duration = len(chunk_audio) / (output_sr if sr != output_sr else sr)
                logger.info(f"[HANDLER] Chunk {idx+1} OK: {duration:.2f}s audio ({chunk_time:.2f}s)")
            else:
                logger.error(f"[HANDLER] Chunk {idx+1} produced no audio")

        except Exception as e:
            logger.error(f"[HANDLER] Chunk {idx+1} error: {e}")
            continue

        # Memory cleanup every 5 chunks
        if (idx + 1) % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    generation_time = time.time() - generation_start

    if full_audio is None or successful_chunks == 0:
        return {
            "error": f"TTS failed. {successful_chunks}/{len(chunks)} chunks succeeded",
            "total_chunks": len(chunks),
            "successful_chunks": successful_chunks
        }

    # Normalize and save
    use_sr = output_sr if sr != output_sr else (sr or TARGET_SR)
    full_audio = normalize_peak(full_audio, peak=0.95)
    final_duration = len(full_audio) / use_sr

    os.makedirs(OUT_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    name = outfile or f"{job_id}.wav"
    out_path = os.path.join(OUT_DIR, name)

    save_start = time.time()
    sf.write(out_path, full_audio, use_sr)
    save_time = time.time() - save_start

    total_time = time.time() - job_start

    logger.info(f"[HANDLER] Saved: {out_path}, duration: {final_duration:.2f}s")
    logger.info(f"[TIMING] Total: {total_time:.2f}s | Preprocess: {preprocess_time:.2f}s | Gen: {generation_time:.2f}s | Save: {save_time:.2f}s")

    result = {
        "job_id": job_id,
        "sample_rate": use_sr,
        "duration": round(final_duration, 2),
        "total_chunks": len(chunks),
        "successful_chunks": successful_chunks,
        "used_prompt_voice": active_prompt_path is not None,
        "inference_mode": inference_mode,
        "model": "CosyVoice2-0.5B",
        "processing_time": round(total_time, 2),
        "generation_time": round(generation_time, 2),
        "preprocess_time": round(preprocess_time, 2)
    }

    if ret_mode == "base64":
        with io.BytesIO() as buf:
            sf.write(buf, full_audio, use_sr, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result["audio_b64"] = b64
    else:
        result["path"] = out_path

    return result

# Start serverless worker
runpod.serverless.start({"handler": handler})
