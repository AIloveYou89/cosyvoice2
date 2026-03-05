# preprocess.py - Vietnamese text preprocessing for CosyVoice2
# Adapted from Vi-SparkTTS preprocess module
# Chunk sizes tuned for CosyVoice2 (handles longer text better than Spark)

import re
from typing import List

try:
    from num2words import num2words
    HAS_NUM2WORDS = True
except ImportError:
    HAS_NUM2WORDS = False

# CosyVoice2 handles longer chunks better than Spark TTS
MIN_CHARS_PER_CHUNK = 60
MAX_CHARS_PER_CHUNK = 180
OPTIMAL_CHUNK_SIZE = 120
PUNCS = r".?!…"

# Cached compiled regex
_number_pattern = re.compile(r"(\d{1,3}(?:\.\d{3})*)(?:\s*(%|[^\W\d_]+))?", re.UNICODE)
_whitespace_pattern = re.compile(r"\s+")
_comma_pattern = re.compile(r"\s*,\s*")
_punct_spacing_pattern = re.compile(r"\s+([,;:])")
_repeated_punct_pattern = re.compile(rf"[{PUNCS}]{{2,}}")
_punct_no_space_pattern = re.compile(rf"([{PUNCS}])(?=\S)")


def normalize_text_vn(text: str) -> str:
    """Normalize Vietnamese text. Không lowercase — CosyVoice2 xử lý mixed case tốt."""
    text = text.strip()
    text = _whitespace_pattern.sub(" ", text)
    text = _comma_pattern.sub(", ", text)

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
    """Thủ thuật dấu . đầu chunk — giúp model đọc ổn định hơn."""
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
    """Main preprocessing: normalize → split → chunk → ensure leading dot."""
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


def get_chunk_stats(chunks: List[str]) -> dict:
    lengths = [len(c) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "optimal_ratio": sum(1 for l in lengths if MIN_CHARS_PER_CHUNK <= l <= OPTIMAL_CHUNK_SIZE) / len(lengths) if lengths else 0
    }


if __name__ == "__main__":
    sample = "Dạ vâng, em hiểu ạ. Cuối tuần này bên em có chương trình tham quan nghỉ dưỡng tại Phú Quốc, hoàn toàn miễn phí ạ. Anh chị có tiện nghe em giới thiệu qua không ạ? Giá trị 10.000.000 đồng."

    print("=== TEST PREPROCESS (CosyVoice2) ===")
    chunks = preprocess_text(sample)
    stats = get_chunk_stats(chunks)

    print(f"Input: {len(sample)} chars")
    print(f"Chunks: {stats['total_chunks']} (range: {stats['min_length']}-{stats['max_length']}, avg: {stats['avg_length']:.1f})")
    print()
    for idx, c in enumerate(chunks, 1):
        print(f"  {idx}. ({len(c)} chars) {repr(c)}")
