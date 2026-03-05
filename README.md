# CosyVoice2 — RunPod Serverless TTS

RunPod serverless handler cho **CosyVoice2-0.5B** (Alibaba FunAudioLLM).
Zero-shot voice clone, streaming 150ms, 9 ngôn ngữ.

## So sánh với SparkTTS

| | SparkTTS 0.5B | CosyVoice2 0.5B |
|---|---|---|
| Tiếng Việt | ✅ Train riêng (Vi-SparkTTS) | ⚠️ Chưa chính thức |
| Streaming | Không | ✅ 150ms |
| Voice clone | consent_audio.wav | Zero-shot (prompt WAV) |
| Sample rate | 24kHz | 22kHz |
| Benchmark SS | 66.0% | 75.7% |

## Files

```
cosyvoice2/
├── rp_handler.py      # RunPod serverless handler
├── preprocess.py       # Vietnamese text preprocessing
├── Dockerfile          # Docker build for RunPod
├── requirements.txt    # Python dependencies
├── test_colab.ipynb    # Colab notebook: CosyVoice2 vs SparkTTS
└── README.md
```

## Quick Start — Colab Test

1. Mở `test_colab.ipynb` trên Google Colab (T4 free)
2. Chạy từ trên xuống
3. So sánh chất lượng tiếng Việt giữa 2 model

## Deploy RunPod

### Build Docker

```bash
docker build -t cosyvoice2-runpod .
```

### RunPod Serverless

1. Push image lên Docker Hub / GHCR
2. Tạo Serverless Endpoint trên RunPod
3. Env vars:
   - `HF_TOKEN` — HuggingFace token
   - `PROMPT_TEXT` — Mô tả giọng mẫu (default: "Tôi là chủ sở hữu giọng nói này.")
4. Network Volume: upload `consent_audio.wav` (giọng Minh) vào `/runpod-volume/`

### API Call

```json
{
  "input": {
    "text": "Dạ, xin chào anh chị ạ.",
    "mode": "zero_shot",
    "prompt_text": "Tôi là chủ sở hữu giọng nói này.",
    "return": "base64"
  }
}
```

### Modes

| Mode | Mô tả | Cần prompt? |
|------|--------|------------|
| `zero_shot` | Voice clone từ prompt WAV | ✅ |
| `cross_lingual` | Clone voice, text khác ngôn ngữ prompt | ✅ |
| `instruct2` | Điều khiển dialect/emotion | ✅ + instruction |

### Instruct2 Example (dialect control)

```json
{
  "input": {
    "text": "Xin chào anh chị.",
    "mode": "instruct2",
    "instruction": "用温柔的语气说<|endofprompt|>",
    "return": "base64"
  }
}
```

## Thủ thuật

- **Dấu `.` đầu chunk** — `". Xin chào."` thay vì `"Xin chào."` → model đọc ổn định hơn
- **Chunk size 60-180 chars** — CosyVoice2 xử lý chunk dài tốt hơn SparkTTS
- **Cross-lingual mode** có thể tốt hơn zero_shot cho tiếng Việt (vì không cần prompt text match)

## Lưu ý

- CosyVoice2 **chưa chính thức hỗ trợ tiếng Việt** — cần test thực tế
- Nếu VN quality kém → thử CosyVoice3 (Fun-CosyVoice3-0.5B-2512) — version mới hơn
- VRAM: ~3-4GB trên T4, đủ chạy song song test
