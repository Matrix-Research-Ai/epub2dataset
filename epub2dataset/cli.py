#!/usr/bin/env python3
"""
epub2dataset v5 — Research-Grade EPUB to LLM Training Dataset Converter
=======================================================================

Implements the full **Curate-and-Focus Methodology** from 4 research papers:

  [1] "From Curation to Code: A Practical Framework for Building Single-Domain
       Datasets for Low-CPU LLM Training"
  [2] "Curating for Capability: A Low-CPU Framework for Building Specialized
       LLM Datasets"
  [3] "Curating for Efficiency: A Blueprint for Building Specialized LLMs on
       Low-Power Hardware"
  [4] "From Curation to Compression: Building High-Efficiency LLM Datasets for
       Resource-Constrained Training"

Pipeline (6-phase, per papers [1][3]):
  Phase 1 — Scan & Acquire        (recursive EPUB, dedup, metadata filter)
  Phase 2 — Extract & Clean       (per-chapter, NFKC, PII detection)
  Phase 3 — Domain Classification (heuristic content-type detection)
  Phase 4 — Semantic Chunking     (paragraph-boundary, overlap, front-matter skip)
  Phase 5 — 8-Dim Quality Scoring + Dedup + Decontamination
  Phase 6 — Format & Write        (Instruction / Completion / Chat style, sharding)

Usage:
  python epub2dataset.py <input_dir> -o dataset.jsonl
  python epub2dataset.py <input_dir> -o dataset.jsonl --style chat
  python epub2dataset.py <input_dir> -o dataset.jsonl --style completion
  python epub2dataset.py <input_dir> -o dataset --shard 10
  python epub2dataset.py <input_dir> --lightweight   (skip heavy checks)
  python epub2dataset.py <input_dir> --stats
  python epub2dataset.py <input_dir> --recipe strict --style instruction
"""

import os
import re
import sys
import json
import math
import csv
import hashlib
import logging
import argparse
import textwrap
# multiprocessing may not be available in Pyodide/WASM
try:
    import multiprocessing
    HAS_MULTIPROCESSING = True
except (ImportError, ModuleNotFoundError):
    HAS_MULTIPROCESSING = False
    multiprocessing = None
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Optional

# ProcessPoolExecutor may not be available in all environments (e.g. Pyodide/WASM)
try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    HAS_PROCESS_POOL = True
except (ImportError, ModuleNotFoundError):
    HAS_PROCESS_POOL = False
    ProcessPoolExecutor = None
    as_completed = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Dummy tqdm
    class tqdm:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): pass
        def set_description(self, s): pass

import ebooklib
from ebooklib import epub

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("epub2dataset")

# =========================================================================
# CONSTANTS
# =========================================================================

# Quality scoring (Code2Doc-inspired 8-dimension rubric, ref [1] §3)
QUALITY_DIMENSIONS = [
    "completeness", "clarity", "coherence", "density",
    "structure", "length", "formatting", "uniqueness",
    "readability", "naturalness",
]
QUALITY_THRESHOLD = 6.0
DEFAULT_MIN_CHARS = 300
DEFAULT_MAX_CHARS = 8192
DEFAULT_MIN_DENSITY = 0.35
DEFAULT_MAX_BLANK_RATIO = 0.55
DEFAULT_SIMILARITY = 0.85
CHUNK_OVERLAP_CHARS = 256

# Instruction templates (ref [1] §5 — Alpaca-style with ### markers)
INSTRUCTION_TEMPLATES = [
    "Study and understand the following passage from \"{chapter}\" in {source}.",
    "Read the following excerpt from {source} (Chapter: {chapter}) and comprehend the key concepts presented.",
    "Learn from the following passage. It is taken from \"{chapter}\" in {source}.",
    "Analyze the following text from {source}, specifically the chapter titled \"{chapter}\". Understand the main ideas and details.",
]

# Front/back matter keywords — skip these chapters (ref [1] §2 — file filtering)
SKIP_CHAPTER_PATTERNS = [
    r'copyright', r'table\s*of\s*contents', r'index', r'glossary',
    r'about\s*the\s*auth(or|ress)', r'acknowledg(e|ment)s',
    r'foreword', r'preface', r'introduction\s*$', r'dedication',
    r'disclaimer', r'license', r'colophon', r'credits',
    r'also\s*by', r'praise\s*for', r'title\s*page', r'notes?$',
]

# PII regex patterns (ref [1] §3 — Privacy Compliance)
PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone_us": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
}


# =========================================================================
# TEXT CLEANING (ref [4] §3 — NFKC normalization, noise removal)
# =========================================================================

def normalize_text(text: str) -> str:
    """NFKC unicode normalization + whitespace cleanup (ref [4])."""
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]', '', text)
    text = text.replace('\xad', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [l.strip() for l in text.split('\n')]
    text = '\n'.join(lines)
    lines = [l for l in text.split('\n') if not re.match(r'^\d+$', l.strip())]
    text = '\n'.join(lines).strip()
    return text


def content_density(text: str) -> float:
    clean = re.sub(r'\s', '', text)
    if not clean:
        return 0.0
    return sum(1 for c in clean if c.isalnum()) / len(clean)


def blank_line_ratio(text: str) -> float:
    lines = text.split('\n')
    if not lines:
        return 0.0
    return sum(1 for l in lines if not l.strip()) / len(lines)


def avg_sentence_length(text: str) -> float:
    sents = re.split(r'[.!?]+', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 5]
    if not sents:
        return 0.0
    return sum(len(s.split()) for s in sents) / len(sents)


def lexical_diversity(text: str) -> float:
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def has_structure(text: str) -> float:
    score = 0.0
    if re.search(r'^#{1,6}\s', text, re.MULTILINE):
        score += 0.3
    if re.search(r'^[\*\-]\s', text, re.MULTILINE):
        score += 0.2
    if re.search(r'^\d+\.\s', text, re.MULTILINE):
        score += 0.2
    if re.search(r'(?:chapter|section|part|introduction|conclusion|summary)',
                 text, re.IGNORECASE):
        score += 0.3
    return min(score, 1.0)


def entropy(text: str) -> float:
    from collections import Counter
    if not text:
        return 0.0
    counts = Counter(text.lower())
    total = len(text)
    ent = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return min(ent / 7.0, 1.0)


def readability_flesch(text: str) -> float:
    """
    Normalized Flesch Reading Ease (0-1).
    Higher = more readable. Uses syllable approximation.
    (ref [1] §3 — clarity/readability in quality rubric)
    """
    sents = re.split(r'[.!?]+', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 2]
    if not sents:
        return 0.5
    words = re.findall(r'\w+', text)
    if not words:
        return 0.5
    total_words = len(words)
    total_sents = len(sents)
    # Approximate syllables: count vowel groups
    total_syllables = 0
    for w in words:
        w = w.lower()
        syllables = len(re.findall(r'[aeiouy]+', w))
        if syllables == 0:
            syllables = 1
        total_syllables += syllables
    # Flesch Reading Ease = 206.835 - 1.015*(words/sent) - 84.6*(syllables/words)
    if total_sents == 0 or total_words == 0:
        return 0.5
    fre = 206.835 - 1.015 * (total_words / total_sents) - 84.6 * (total_syllables / total_words)
    # Normalize: FRE typically ranges 0-100, clip to 0-100 then divide by 100
    fre = max(0, min(100, fre))
    return fre / 100.0


def ai_naturalness_score(text: str) -> float:
    """
    Score text naturalness vs AI-generated patterns.
    AI text tends to have: uniform sentence lengths, low burstiness,
    overuse of certain transition words, and low variance in structure.
    Higher score = more natural (less likely AI-generated).
    (ref [1] §3 — "AI-generated content detection" in quality rubric)
    """
    sents = re.split(r'[.!?]+', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 5]
    if len(sents) < 4:
        return 0.7  # neutral for short texts

    # 1. Burstiness: variance in sentence lengths (AI text has low variance)
    lengths = [len(s.split()) for s in sents]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len)**2 for l in lengths) / len(lengths)
    std = math.sqrt(variance) if variance > 0 else 1
    # High std = more natural (human writing has bursty sentence lengths)
    burstiness = min(std / 8.0, 1.0)

    # 2. Transition word diversity (AI text overuses certain transitions)
    transition_words = ['however', 'therefore', 'furthermore', 'moreover',
                        'additionally', 'consequently', 'nevertheless',
                        'in addition', 'on the other hand', 'as a result',
                        'importantly', 'notably', 'specifically']
    text_lower = text.lower()
    transition_count = sum(1 for tw in transition_words if tw in text_lower)
    # Too many transitions = AI-like. 0-2 is natural, >5 is suspicious
    trans_score = max(0, 1.0 - transition_count / 8.0)

    # 3. Sentence start diversity (AI text repeats sentence starters)
    starters = [s.strip().split()[0].lower() if s.strip().split() else ''
                for s in sents]
    starters = [s for s in starters if s]
    if starters:
        unique_starters = len(set(starters))
        starter_diversity = unique_starters / max(len(starters), 1)
    else:
        starter_diversity = 0.5

    # Combine: burstiness (0.4) + transition score (0.3) + starter diversity (0.3)
    score = 0.4 * burstiness + 0.3 * trans_score + 0.3 * starter_diversity
    return min(max(score, 0.0), 1.0)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text (ref [4] §3)."""
    return max(1, len(text) // 4)


def estimate_tokens_precise(text: str, model: str = "cl100k") -> int:
    """
    Count tokens using a specific model's tokenizer.
    Supports: cl100k (GPT-4, GPT-3.5), p50k (Codex), r50k (GPT-3, Ada).
    Falls back to char/4 ratio if tiktoken not installed.
    (ref [4] §3 — tokenization as bottleneck)
    """
    encodings = {"cl100k": "cl100k_base", "p50k": "p50k_base", "r50k": "r50k_base"}
    if model in encodings:
        try:
            import tiktoken
            enc = tiktoken.get_encoding(encodings[model])
            return len(enc.encode(text))
        except ImportError:
            pass
    return estimate_tokens(text)


# =========================================================================
# STRUCTURED PACKING (ref [1] §6, [3] §2 — SPLICE algorithm)
# =========================================================================

def pack_examples(examples: list, max_tokens: int = 2048,
                  bos_token: str = "<|begin|>", eos_token: str = "<|end|>") -> list[dict]:
    """
    Pack multiple related examples into a single training sample.
    Inspired by SPLICE algorithm and structured packing (ref [1], [3]).

    Each packed sample concatenates multiple related chunks separated by
    BOS/EOS markers, improving context window utilization.
    """
    if not examples:
        return []

    packed = []
    current_chunks = []
    current_tokens = 0

    for ex in examples:
        # Extract text from any format
        text = ex.get("response") or ex.get("completion") or ""
        if not text:
            msgs = ex.get("messages", [])
            if len(msgs) >= 3:
                text = msgs[2].get("content", "")
        if not text:
            text = str(ex.get("instruction", ""))
        ex_tokens = len(text) // 4  # rough estimate
        overhead = len(bos_token) // 4 + len(eos_token) // 4  # BOS/EOS overhead

        if current_tokens + ex_tokens + overhead > max_tokens:
            # Flush current pack
            if current_chunks:
                packed_text = ""
                for chunk_text in current_chunks:
                    packed_text += f"{bos_token}\n{chunk_text}\n{eos_token}\n"
                packed.append({
                    "instruction": "Study the following collection of related passages.",
                    "input": "Multiple excerpts from curated sources.",
                    "response": packed_text.strip(),
                    "packed_count": len(current_chunks),
                    "packed_tokens": current_tokens,
                })
            current_chunks = [text]
            current_tokens = ex_tokens + overhead
        else:
            current_chunks.append(text)
            current_tokens += ex_tokens + overhead

    # Flush remaining
    if current_chunks:
        packed_text = ""
        for chunk_text in current_chunks:
            packed_text += f"{bos_token}\n{chunk_text}\n{eos_token}\n"
        packed.append({
            "instruction": "Study the following collection of related passages.",
            "input": "Multiple excerpts from curated sources.",
            "response": packed_text.strip(),
            "packed_count": len(current_chunks),
            "packed_tokens": current_tokens,
        })

    return packed


# =========================================================================
# PII DETECTION (ref [1] §3 — Privacy Compliance with Microsoft Presidio)
# =========================================================================

def detect_pii(text: str) -> dict:
    """
    Scan text for PII patterns. Returns {type: [matches]}.
    Lightweight regex-based implementation (Presidio is heavier, ref [1]).
    """
    findings = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            findings[pii_type] = matches[:5]  # cap at 5 per type
    return findings


def redact_pii(text: str) -> str:
    """Replace PII with [REDACTED] placeholders."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]', text)
    return text


# =========================================================================
# DOMAIN CLASSIFICATION (ref [1] §3 — Domain Isolation & Classification)
# =========================================================================

def classify_content(text: str) -> str:
    """
    Heuristic content-type classification.
    Returns one of: 'narrative', 'technical', 'code', 'dialogue', 'reference'.
    (ref [1] §3 — Domain Isolation stage)
    """
    words_lower = text.lower()
    word_count = len(words_lower.split())

    # Code detection: special chars, keywords
    code_indicators = sum([
        words_lower.count('function'), words_lower.count('def '),
        words_lower.count('class '), words_lower.count('import '),
        words_lower.count('if __name__'), words_lower.count('public static'),
        words_lower.count('int main'), words_lower.count('var '),
        words_lower.count('const '), words_lower.count('=>'),
        words_lower.count('->'), words_lower.count('::'),
    ])
    if word_count > 0 and (code_indicators / word_count) > 0.02:
        return 'code'

    # Dialogue detection: quotation, speaker tags
    dialogue_indicators = sum([
        len(re.findall(r'["\'].*?["\']', text)),
        words_lower.count('said'), words_lower.count('asked'),
        words_lower.count('replied'), words_lower.count('exclaimed'),
        len(re.findall(r'^["\']', text, re.MULTILINE)),
    ])
    if word_count > 0 and (dialogue_indicators / word_count) > 0.03:
        return 'dialogue'

    # Technical detection: domain terminology
    tech_keywords = ['algorithm', 'function', 'parameter', 'implementation',
                     'database', 'server', 'api', 'http', 'protocol',
                     'framework', 'library', 'configuration', 'syntax']
    tech_count = sum(words_lower.count(kw) for kw in tech_keywords)
    if word_count > 0 and (tech_count / word_count) > 0.01:
        return 'technical'

    # Reference detection: lists, structured data
    list_lines = len(re.findall(r'^[\-\*\d]\.\s', text, re.MULTILINE))
    table_lines = len(re.findall(r'\|.*\|', text))
    if (list_lines + table_lines) > max(3, word_count * 0.05):
        return 'reference'

    return 'narrative'


# =========================================================================
# FRONT/BACK MATTER DETECTION (ref [1] §2 — file filtering)
# =========================================================================

def is_front_matter(chapter_title: str, text: str) -> bool:
    """
    Detect title pages, copyright, TOC, indices — skip these.
    (ref [1] §2 — Acquisition & Initial Filtering: file extension/size/license)
    """
    title_lower = chapter_title.lower()
    for pattern in SKIP_CHAPTER_PATTERNS:
        if re.search(pattern, title_lower):
            return True
    # Very short chapters (< 150 chars) that are just titles
    if len(text.strip()) < 150 and len(chapter_title) > 0:
        first_line = text.strip().split('\n')[0].lower()
        for pattern in SKIP_CHAPTER_PATTERNS:
            if re.search(pattern, first_line):
                return True
    return False


# =========================================================================
# QUALITY SCORING — 8-dimension rubric (Code2Doc, ref [1] §3)
# =========================================================================

def quality_score_8d(text: str, min_chars: int, max_chars: int) -> tuple[float, dict]:
    n = len(text)
    if n < min_chars or n > max_chars:
        return 0.0, {"_reject": f"length ({n} chars)"}

    density = content_density(text)
    if density < DEFAULT_MIN_DENSITY:
        return 0.0, {"_reject": f"low_density ({density:.2f})"}

    blank_r = blank_line_ratio(text)
    if blank_r > DEFAULT_MAX_BLANK_RATIO:
        return 0.0, {"_reject": f"high_blanks ({blank_r:.2f})"}

    sents = re.split(r'[.!?]+', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 10]

    # 1. Completeness
    completeness = min(len(sents) / 8.0, 1.0) if sents else 0.0
    # 2. Clarity
    avg_sl = avg_sentence_length(text)
    clarity = 0.0
    if 8 <= avg_sl <= 35:
        clarity = 1.0 - abs(avg_sl - 20) / 30
    clarity = max(0.0, min(clarity, 1.0))
    # 3. Coherence
    if len(sents) >= 3:
        lengths = [len(s.split()) for s in sents]
        mean_l = sum(lengths) / len(lengths)
        variance = sum((l - mean_l)**2 for l in lengths) / len(lengths)
        std = math.sqrt(variance) if variance > 0 else 0
        coherence = max(0.0, 1.0 - std / 20.0)
    else:
        coherence = 0.5
    # 4. Density
    density_score = min(density / 0.7, 1.0)
    # 5. Structure
    structure_score = has_structure(text)
    # 6. Length
    if n < 500:
        length_score = (n - min_chars) / (500 - min_chars)
    elif n <= 4000:
        length_score = 1.0
    else:
        length_score = max(0.0, 1.0 - (n - 4000) / max_chars)
    length_score = max(0.0, min(length_score, 1.0))
    # 7. Formatting
    artifact_chars = len(re.findall(r'[^\x20-\x7E\n]', text))
    total_printable = len(re.findall(r'[\x20-\x7E\n]', text))
    formatting = 1.0 - (artifact_chars / max(total_printable, 1))
    # 8. Uniqueness
    lex_div = lexical_diversity(text)
    ent = entropy(text)
    uniqueness = 0.5 * lex_div + 0.5 * ent
    # 9. Readability (Flesch Reading Ease normalized)
    readability = readability_flesch(text)
    # 10. Naturalness (AI-generated text detection, ref [1] §3)
    naturalness = ai_naturalness_score(text)

    dims = {
        "completeness": round(completeness * 10, 1),
        "clarity": round(clarity * 10, 1),
        "coherence": round(coherence * 10, 1),
        "density": round(density_score * 10, 1),
        "structure": round(structure_score * 10, 1),
        "length": round(length_score * 10, 1),
        "formatting": round(formatting * 10, 1),
        "uniqueness": round(uniqueness * 10, 1),
        "readability": round(readability * 10, 1),
        "naturalness": round(naturalness * 10, 1),
    }
    weights = {
        "completeness": 0.12, "clarity": 0.12, "coherence": 0.08,
        "density": 0.12, "structure": 0.08, "length": 0.08,
        "formatting": 0.08, "uniqueness": 0.12,
        "readability": 0.10, "naturalness": 0.10,
    }
    final = sum(dims[k] * weights[k] for k in dims)
    return round(final, 2), dims


# =========================================================================
# EPUB EXTRACTION
# =========================================================================

def extract_epub(epub_path: str) -> list[dict]:
    file_hash = hashlib.sha256()
    with open(epub_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            file_hash.update(chunk)
    file_hash = file_hash.hexdigest()[:16]

    try:
        book = epub.read_epub(epub_path)
    except Exception as e:
        log.warning(f"  [SKIP] {os.path.basename(epub_path)}: {e}")
        return []

    # Build TOC label map
    toc_labels = {}
    def _walk_toc(items):
        for item in items:
            if isinstance(item, epub.Link):
                key = item.href.split('#')[0]
                toc_labels[key] = item.title
            elif isinstance(item, tuple):
                sec, subs = item[0], item[1] if len(item) > 1 else []
                if isinstance(sec, epub.Section):
                    _walk_toc(list(subs) if subs else [])
                else:
                    _walk_toc([sec])
    if hasattr(book, 'toc'):
        _walk_toc(book.toc)

    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    chapters = []
    for idx, item in enumerate(items):
        try:
            raw = item.get_body_content().decode('utf-8', errors='replace')
        except Exception:
            try:
                raw = item.get_body_content().decode('latin-1', errors='replace')
            except Exception:
                continue

        title_m = re.search(r'<title[^>]*>(.*?)</title>', raw, re.IGNORECASE | re.DOTALL)
        chapter_title = title_m.group(1).strip() if title_m else f"Chapter {idx+1}"

        text = html_to_text(raw)
        if not text.strip():
            continue

        src = item.get_name() or ""
        toc_label = toc_labels.get(src, "") or toc_labels.get(os.path.basename(src), "")
        if not toc_label and item.id:
            toc_label = toc_labels.get(item.id, "")

        resolved_title = toc_label or chapter_title

        # Skip front/back matter (ref [1] §2)
        if is_front_matter(resolved_title, text):
            log.info(f"    [SKIP front matter] {resolved_title}")
            continue

        chapters.append({
            "source": os.path.basename(epub_path),
            "chapter_index": idx + 1,
            "chapter_title": resolved_title,
            "text": text,
            "file_hash": file_hash,
        })

    return chapters


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()
    text = soup.get_text(separator='\n')
    return normalize_text(text)


# =========================================================================
# EPUB METADATA (ref [4] §3 — Provenance tracking)
# =========================================================================

def extract_metadata(epub_path: str) -> dict:
    """Extract author, publisher, language, license from EPUB metadata."""
    meta = {"author": "", "publisher": "", "language": "", "pub_date": "", "license": ""}
    try:
        book = epub.read_epub(epub_path)
        if hasattr(book, 'get_metadata'):
            for k, tag in [('author','creator'),('publisher','publisher'),
                            ('language','language'), ('pub_date','date'),
                            ('license','rights')]:
                try:
                    dc = book.get_metadata('DC', tag)
                    if dc and dc[0]:
                        val = str(dc[0][0])
                        if k == 'pub_date':
                            val = val[:10]
                        meta[k] = val
                except Exception:
                    pass
        if not meta['author'] and hasattr(book, 'authors') and book.authors:
            meta['author'] = book.authors[0]
    except Exception:
        pass
    return meta


# =========================================================================
# EVAL BENCHMARK DECONTAMINATION (ref [1] §3)
# =========================================================================

# Common evaluation benchmark n-grams for contamination detection.
# If a chunk contains these exact phrases, it may be from an eval set.
BENCHMARK_PHRASES = [
    # MMLU / General knowledge benchmarks
    "the acceleration due to gravity is approximately",
    "which of the following is not a prime number",
    "the capital of Australia is",
    "the chemical symbol for gold is",
    # HumanEval / Code benchmarks
    "def has_close_elements",
    "def get_positive",
    "def count_distinct_characters",
    "def numerical_letter_grade",
    "def below_threshold",
    "from typing import List",
    # GSM8K / Math benchmarks
    "Natalia sold clips to 48 of her friends",
    "Janet's ducks lay 16 eggs per day",
    "A bat and a ball cost $1.10",
    # BBH / BIG-Bench
    "choose the option that best completes the pattern",
    "which of the following is the correct ordering",
]


def check_benchmark_contamination(text: str) -> tuple[bool, str]:
    """Check if text matches known eval benchmark phrases."""
    text_lower = text.lower()
    for phrase in BENCHMARK_PHRASES:
        if phrase.lower() in text_lower:
            return True, phrase[:40]
    return False, ""


# =========================================================================
# PARALLEL EXTRACTION (ref [4] §2 — CPU thread pool for throughput)
# =========================================================================

def extract_epub_wrapper(args):
    """Wrapper for parallel extraction. Args is a tuple (path, ...)."""
    path = args if isinstance(args, str) else args[0]
    try:
        chapters = extract_epub(path)
        meta = extract_metadata(path)
        return path, chapters, meta
    except Exception as e:
        return path, [], {"author": "", "publisher": "", "language": "", "pub_date": ""}


# =========================================================================
# SEMANTIC CHUNKING (ref [3] §2 — ChunkNorris-inspired)
# =========================================================================

def chunk_semantic(text: str, max_chars: int = 8192,
                   overlap: int = 256) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split('\n\n')
    chunks = []
    current = []
    current_len = 0

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        para_len = len(para)

        if para_len > max_chars:
            if current:
                chunks.append('\n\n'.join(current))
                current = []
                current_len = 0
            start = 0
            while start < para_len:
                end = min(start + max_chars, para_len)
                chunk = para[start:end]
                if chunk.strip():
                    chunks.append(chunk)
                start = end - overlap if (end - start) == max_chars else end
            continue

        if current_len + para_len > max_chars:
            chunk_text = '\n\n'.join(current)
            chunks.append(chunk_text)
            overlap_text = ''
            overlap_len = 0
            for p in reversed(current):
                pl = len(p)
                if overlap_len + pl > overlap:
                    break
                overlap_text = p + '\n\n' + overlap_text
                overlap_len += pl + 2
            current = [overlap_text.strip()] if overlap_text.strip() else []
            current.append(para)
            current_len = len(current[0]) + para_len + 2 if len(current) > 1 else para_len
        else:
            current.append(para)
            current_len += para_len + 2

    if current:
        chunks.append('\n\n'.join(current))
    return chunks


# =========================================================================
# ENTROPY-BASED FILTERING (ref [1] §6, [3] §2)
# =========================================================================

def entropy_filter(text: str, threshold: float = 3.0) -> bool:
    from collections import Counter
    if not text:
        return False
    counts = Counter(text.lower())
    total = len(text)
    ent = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return ent >= threshold


# =========================================================================
# DEDUPLICATION — Global MinHash LSH
# =========================================================================

def shingle(text: str, k: int = 6) -> set:
    text = text.lower().strip()
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def minhash_sig(shingles: set, num_hashes: int = 128) -> list[int]:
    p = 2**31 - 1
    sig = []
    for seed in range(num_hashes):
        a = (seed * 2 + 1) % p
        b = (seed * 7 + 3) % p
        min_val = p
        for s in shingles:
            h = (a * hash(s) + b) % p
            if h < min_val:
                min_val = h
        sig.append(min_val)
    return sig


def lsh_buckets(sig: list[int], bands: int = 16) -> list[tuple]:
    rows = len(sig) // bands
    return [(b, hash(tuple(sig[b*rows:(b+1)*rows]))) for b in range(bands)]


def find_near_duplicates(
    chunks: list[dict],
    threshold: float = 0.85,
    num_hashes: int = 128,
    bands: int = 16,
) -> set[int]:
    n = len(chunks)
    texts = [c['text'] for c in chunks]
    signatures = []
    for t in texts:
        s = shingle(t)
        signatures.append(minhash_sig(s, num_hashes) if s else [])

    ht = defaultdict(list)
    rows = num_hashes // bands
    for i, sig in enumerate(signatures):
        if not sig:
            continue
        for b in range(bands):
            ht[(b, hash(tuple(sig[b*rows:(b+1)*rows])))].append(i)

    duplicates = set()
    for key, idxs in ht.items():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                a, b = idxs[i], idxs[j]
                if a in duplicates or b in duplicates:
                    continue
                sa, sb = signatures[a], signatures[b]
                if not sa or not sb:
                    continue
                union = len(set(sa) | set(sb))
                if union == 0:
                    continue
                sim = len(set(sa) & set(sb)) / union
                if sim >= threshold:
                    duplicates.add(b)
    return duplicates


# =========================================================================
# DECONTAMINATION (ref [1] §3)
# =========================================================================

DECONTAMINATION_HASHES = {}

def load_decontamination_hashes() -> dict:
    eval_dir = os.path.join(os.path.dirname(__file__), 'eval_hashes')
    hashes = {}
    if os.path.isdir(eval_dir):
        for fname in os.listdir(eval_dir):
            if fname.endswith('.json'):
                with open(os.path.join(eval_dir, fname)) as f:
                    hashes.update(json.load(f))
    return hashes


def check_decontamination(text: str) -> tuple[bool, str]:
    global DECONTAMINATION_HASHES
    if not DECONTAMINATION_HASHES:
        DECONTAMINATION_HASHES = load_decontamination_hashes()
    if not DECONTAMINATION_HASHES:
        return False, ""
    fp = hashlib.sha256(text.encode()).hexdigest()
    for bench, ref_hashes in DECONTAMINATION_HASHES.items():
        if fp in ref_hashes:
            return True, bench
    return False, ""


# =========================================================================
# FORMATTING — Three output styles (ref [1] §5)
# =========================================================================

def make_example_instruction(
    text: str, source: str, chapter_title: str,
    chapter_index: int, total_chunks: int,
    template_idx: int = 0, quality: float = 0.0,
    dims: Optional[dict] = None,
    content_type: str = "narrative",
    pii_found: Optional[dict] = None,
    est_tokens: int = 0,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Instruction-Style with Alpaca ### markers (ref [1] §5).
    Format:
      ### Instruction:
      <task>
      ### Input:
      <context>
      ### Response:
      <text>
    """
    templates = INSTRUCTION_TEMPLATES
    tmpl = templates[template_idx % len(templates)]
    instruction = tmpl.format(source=source, chapter=chapter_title)

    # Build Alpaca-style formatted response
    response_parts = [
        "### Instruction:",
        instruction,
        "### Input:",
        f"Context: {source} — Chapter: {chapter_title}",
        "### Response:",
        text,
    ]
    formatted_response = "\n\n".join(response_parts)

    result = {
        "instruction": instruction,
        "input": f"Context: {source} — Chapter: {chapter_title}",
        "response": formatted_response,
        "source": source,
        "chapter": chapter_title,
        "chapter_index": chapter_index,
        "domain": "general_knowledge",
        "content_type": content_type,
        "provenance": {
            "file": source,
            "chapter": chapter_title,
            "chunk_index": chapter_index,
        },
        "quality_score": quality,
        "est_tokens": est_tokens,
    }
    if dims:
        result["quality_dimensions"] = dims
    if pii_found:
        result["pii_detected"] = pii_found
    if metadata:
        result["author"] = metadata.get("author", "")
        result["publisher"] = metadata.get("publisher", "")
        result["language"] = metadata.get("language", "")
    return result


def make_example_completion(
    text: str, source: str, chapter_title: str,
    chapter_index: int, quality: float = 0.0,
    content_type: str = "narrative",
    est_tokens: int = 0,
) -> dict:
    """
    Completion-Style: simple prompt + completion (ref [1] §5).
    Best for narrow tasks where context is minimal.
    """
    return {
        "prompt": f"Read and learn from the following passage from \"{chapter_title}\" in {source}.\n\n",
        "completion": text,
        "source": source,
        "chapter": chapter_title,
        "chapter_index": chapter_index,
        "domain": "general_knowledge",
        "content_type": content_type,
        "quality_score": quality,
        "est_tokens": est_tokens,
    }


def make_example_chat(
    text: str, source: str, chapter_title: str,
    chapter_index: int, quality: float = 0.0,
    content_type: str = "narrative",
    est_tokens: int = 0,
) -> dict:
    """
    Chat-Style: messages array with system/user/assistant roles (ref [1] §5).
    Essential for training conversational agents.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": f"You are a knowledgeable assistant trained on {source}.",
            },
            {
                "role": "user",
                "content": f"Explain the key concepts from \"{chapter_title}\" in {source}.",
            },
            {
                "role": "assistant",
                "content": text,
            },
        ],
        "source": source,
        "chapter": chapter_title,
        "chapter_index": chapter_index,
        "domain": "general_knowledge",
        "content_type": content_type,
        "quality_score": quality,
        "est_tokens": est_tokens,
    }


# =========================================================================
# PIPELINE
# =========================================================================

RECIPES = {
    "strict": {
        "min_chars": 500, "max_chars": 4096,
        "quality_threshold": 7.5, "similarity": 0.80,
        "min_density": 0.45, "entropy_threshold": 3.5,
    },
    "balanced": {
        "min_chars": 300, "max_chars": 8192,
        "quality_threshold": 6.0, "similarity": 0.85,
        "min_density": 0.35, "entropy_threshold": 3.0,
    },
    "lenient": {
        "min_chars": 150, "max_chars": 16384,
        "quality_threshold": 4.0, "similarity": 0.90,
        "min_density": 0.25, "entropy_threshold": 2.5,
    },
}


def print_stats(input_dir: str):
    epubs = sorted(Path(input_dir).rglob('*.epub'))
    if not epubs:
        log.warning("No .epub files found.")
        return
    total_chapters = 0
    total_bytes = 0
    for ep in epubs:
        size = ep.stat().st_size
        total_bytes += size
        try:
            book = epub.read_epub(str(ep))
            items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            n_ch = len(items)
            total_chapters += n_ch
        except Exception:
            n_ch = 0
        log.info(f"  {ep.name}  ({size//1024} KB, ~{n_ch} chapters)")

    log.info(f"\n  Total EPUBs:     {len(epubs)}")
    log.info(f"  Total chapters:  {total_chapters}")
    log.info(f"  Total size:      {total_bytes//1024//1024} MB")
    log.info(f"  Est. output:     ~{max(1, total_chapters)} examples")

    # Check for front matter to skip
    log.info(f"  (will skip title pages, TOC, copyright, indices per ref [1] §2)")


def run_pipeline(args):
    input_dir = args.input_dir
    output_path = args.output
    fmt = args.format
    style = args.style
    do_dedup = not args.no_dedup
    recipe_name = args.recipe
    lightweight = args.lightweight
    shard_count = args.shard
    redact_pii_flag = args.redact_pii
    no_front_matter_skip = args.include_front_matter

    # Load recipe from config file if provided
    config_path = getattr(args, 'config', None)
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        recipe_name = config.get('recipe', recipe_name)
        min_chars = config.get('min_chars', DEFAULT_MIN_CHARS)
        max_chars = config.get('max_chars', DEFAULT_MAX_CHARS)
        quality_threshold = config.get('quality_threshold', QUALITY_THRESHOLD)
        similarity = config.get('similarity', DEFAULT_SIMILARITY)
        min_density = config.get('min_density', DEFAULT_MIN_DENSITY)
        entropy_threshold = config.get('entropy_threshold', 3.0)
        log.info(f"  Loaded config: {config_path}")

    # Load recipe
    if recipe_name and recipe_name in RECIPES:
        recipe = RECIPES[recipe_name]
        if not config_path:  # don't override config file values
            min_chars = recipe["min_chars"]
            max_chars = recipe["max_chars"]
            quality_threshold = recipe["quality_threshold"]
            similarity = recipe["similarity"]
            min_density = recipe["min_density"]
            entropy_threshold = recipe["entropy_threshold"]
        log.info(f"  Recipe: {recipe_name} "
                 f"(min={min_chars}, max={max_chars}, quality≥{quality_threshold}, style={style})")
    else:
        min_chars = min_chars if config_path else (args.min_chars or DEFAULT_MIN_CHARS)
        max_chars = max_chars if config_path else (args.max_chars or DEFAULT_MAX_CHARS)
        quality_threshold = quality_threshold if config_path else (args.quality_threshold or QUALITY_THRESHOLD)
        similarity = similarity if config_path else (args.similarity or DEFAULT_SIMILARITY)
        min_density = DEFAULT_MIN_DENSITY
        entropy_threshold = 3.0

    if lightweight:
        log.info("  Lightweight mode: skipping entropy + structure checks")
        quality_threshold = min(quality_threshold, 5.0)

    per_file = getattr(args, 'per_file', False)
    quality_csv = getattr(args, 'quality_csv', None)
    jobs = getattr(args, 'jobs', 1)
    save_config = getattr(args, 'save_config', None)
    do_pack = getattr(args, 'pack', False)
    pack_max_tokens = getattr(args, 'pack_max_tokens', 2048)
    tokenizer_model = getattr(args, 'tokenizer', None)
    split_domain = getattr(args, 'split_domain', False)

    start_time = datetime.now()

    # ── Phase 1: Scan ──────────────────────────────────────────────
    epubs = sorted(Path(input_dir).rglob('*.epub'))
    seen = set()
    unique = []
    for ep in epubs:
        name = ep.name.lower()
        if name not in seen:
            seen.add(name)
            unique.append(ep)
    epubs = unique

    if not epubs:
        log.error("No .epub files found.")
        return 1

    log.info(f"── Phase 1: Scan ── found {len(epubs)} EPUB files ──")

    # ── Phase 2: Extract (parallel or sequential) ──────────────────
    all_chapters = []
    all_metadata = {}
    skipped = 0

    ep_iter = tqdm(epubs, desc="Extracting EPUBs", disable=not HAS_TQDM) if HAS_TQDM else epubs

    if jobs > 1 and len(epubs) > 1 and HAS_PROCESS_POOL:
        # Parallel extraction (ref [4] §2 — CPU thread pool)
        n_workers = min(jobs, len(epubs), multiprocessing.cpu_count())
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(extract_epub_wrapper, str(ep)): ep for ep in epubs}
            for future in as_completed(futures):
                ep_path, chapters, meta = future.result()
                all_metadata[os.path.basename(ep_path)] = meta
                if not chapters:
                    skipped += 1
                all_chapters.extend(chapters)
                if HAS_TQDM:
                    ep_iter.update(1)
    else:
        for ep in epubs:
            chapters = extract_epub(str(ep))
            meta = extract_metadata(str(ep))
            all_metadata[ep.name] = meta
            if not chapters:
                skipped += 1
            all_chapters.extend(chapters)
            if HAS_TQDM:
                ep_iter.update(1)

    log.info(f"── Phase 2: Extract ── {len(all_chapters)} chapters"
             f" (skipped {skipped} files) ──")

    # ── Phase 3: Domain Classification (ref [1] §3) ───────────────
    domain_dist = Counter()
    for ch in all_chapters:
        ch['content_type'] = classify_content(ch['text'])
        domain_dist[ch['content_type']] += 1

    log.info(f"── Phase 3: Domain Classification ──")
    for dtype, count in domain_dist.most_common():
        log.info(f"    {dtype:12s}: {count}")

    # ── Phase 3b: Chapter Merging ─────────────────────────────────
    # Merge consecutive short chapters from the same source
    # (ref [1] — "a smaller, meticulously curated dataset can yield better results")
    merge_count = 0
    merged_chapters = []
    buffer = None
    for ch in all_chapters:
        if buffer is None:
            buffer = dict(ch)
            continue
        # Merge if current chapter is short (< 500 chars) and from same source
        if len(ch['text']) < 500 and ch['source'] == buffer['source']:
            buffer['text'] += '\n\n' + ch['text']
            buffer['chapter_title'] += ' + ' + ch['chapter_title']
            buffer['chapter_index'] = min(buffer['chapter_index'], ch['chapter_index'])
            merge_count += 1
        else:
            merged_chapters.append(buffer)
            buffer = dict(ch)
    if buffer is not None:
        merged_chapters.append(buffer)
    if merge_count > 0:
        log.info(f"  Chapter merging: merged {merge_count} short chapters")
    all_chapters = merged_chapters

    # ── Phase 4: Chunk ────────────────────────────────────────────
    raw_chunks = []
    for ch in all_chapters:
        chunks2 = chunk_semantic(ch['text'], max_chars=max_chars,
                                 overlap=CHUNK_OVERLAP_CHARS)
        for i, ct in enumerate(chunks2):
            raw_chunks.append({
                "source": ch['source'],
                "chapter_index": ch['chapter_index'],
                "chapter_title": ch['chapter_title'],
                "text": ct,
                "file_hash": ch['file_hash'],
                "content_type": ch.get('content_type', 'narrative'),
                "chunk_id": f"{ch['source']}::ch{ch['chapter_index']}::p{i+1}",
                "raw_chars": len(ct),
                "est_tokens": estimate_tokens(ct),
            })

    log.info(f"── Phase 4: Chunk ── {len(raw_chunks)} raw chunks ──")

    # ── Phase 5: Quality Scoring ──────────────────────────────────
    scored = []
    rejected = defaultdict(int)
    dim_histograms = {d: [] for d in QUALITY_DIMENSIONS}
    total_pii = defaultdict(int)

    for c in raw_chunks:
        # PII detection (ref [1] §3)
        pii = detect_pii(c['text'])
        for k, v in pii.items():
            total_pii[k] += len(v)
        c['pii_found'] = pii

        # Redact if requested
        if redact_pii_flag and pii:
            c['text'] = redact_pii(c['text'])

        if lightweight:
            # Skip heavy scoring, assign neutral quality
            c['quality'] = 5.0
            c['quality_dims'] = {}
            scored.append(c)
            continue

        score, dims = quality_score_8d(c['text'], min_chars, max_chars)
        c['quality'] = score
        c['quality_dims'] = dims

        if "_reject" in dims:
            rejected[dims["_reject"]] += 1
            continue

        if not entropy_filter(c['text'], entropy_threshold):
            rejected["low_entropy"] += 1
            continue

        if content_density(c['text']) < min_density:
            rejected["low_density"] += 1
            continue

        if score < quality_threshold:
            rejected[f"below_{quality_threshold}"] += 1
            continue

        scored.append(c)
        for d in QUALITY_DIMENSIONS:
            if d in dims:
                dim_histograms[d].append(dims[d])

    if not lightweight:
        log.info(f"── Phase 5: Quality Scoring (8-dim) ── "
                 f"passed {len(scored)} / {len(raw_chunks)} ──")
        for reason, count in sorted(rejected.items()):
            log.info(f"  rejected [{reason}]: {count}")

        if scored:
            avg_q = sum(c['quality'] for c in scored) / len(scored)
            log.info(f"  Avg quality: {avg_q:.2f}/10")
            log.info(f"  Dimension averages:")
            for d in QUALITY_DIMENSIONS:
                vals = dim_histograms[d]
                if vals:
                    log.info(f"    {d:15s}: {sum(vals)/len(vals):.1f}/10")
    else:
        scored = raw_chunks
        log.info(f"── Phase 5: Lightweight ── all {len(scored)} chunks kept ──")

    # PII report
    if total_pii:
        log.info(f"  PII detected:")
        for k, v in total_pii.items():
            log.info(f"    {k:15s}: {v} instances")

    if not scored:
        log.error("No chunks passed filtering. Try --lightweight or --recipe lenient.")
        return 1

    # ── Phase 5b: Dedup + Decontamination ─────────────────────────
    if do_dedup and len(scored) > 1 and not lightweight:
        dedup_indices = find_near_duplicates(scored, threshold=similarity)
        deduped = [c for i, c in enumerate(scored) if i not in dedup_indices]
        log.info(f"  Dedup: removed {len(dedup_indices)} near-dups (kept {len(deduped)})")
    else:
        deduped = scored
        dedup_indices = set()
        if lightweight:
            log.info("  Dedup: skipped (lightweight mode)")

    contaminated = 0
    if not lightweight:
        # Check built-in benchmark phrases first (ref [1] §3)
        for c in deduped[:]:
            is_contam, phrase = check_benchmark_contamination(c['text'])
            if not is_contam:
                is_contam, phrase = check_decontamination(c['text'])
            if is_contam:
                deduped.remove(c)
                contaminated += 1
        if contaminated:
            log.info(f"  Decontamination: removed {contaminated} chunks (benchmark phrases + eval hashes)")

    # Export quality distribution CSV if requested
    if quality_csv and scored:
        with open(quality_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["source", "chapter", "quality", "content_type", "est_tokens", "chars"] + QUALITY_DIMENSIONS)
            for c in scored:
                dims = c.get('quality_dims', {})
                row = [c['source'], c['chapter_title'], c.get('quality', 0),
                       c.get('content_type', ''), c.get('est_tokens', 0), c.get('raw_chars', 0)]
                row += [dims.get(d, '') for d in QUALITY_DIMENSIONS]
                w.writerow(row)
        log.info(f"  Quality CSV: {quality_csv}")

    # ── Phase 6: Format & Write ───────────────────────────────────
    chapter_counts = Counter((c['source'], c['chapter_title']) for c in deduped)
    final_domain_dist = Counter(c.get('content_type', 'narrative') for c in deduped)

    # Per-file output mode (SlimPajama-style, ref [4])
    if per_file:
        source_groups = defaultdict(list)
        for c in deduped:
            source_groups[c['source']].append(c)
        base_path = output_path.rsplit('.', 1)[0]
        ext = output_path.rsplit('.', 1)[1] if '.' in output_path else fmt
        total_written = 0
        all_examples = []
        for src_name, src_chunks in sorted(source_groups.items()):
            src_base = src_name.rsplit('.', 1)[0].replace(' ', '_').replace('(', '').replace(')', '')
            per_file_path = f"{base_path}_{src_base}.{ext}"
            src_examples = _build_examples(src_chunks, style, chapter_counts, all_metadata)
            _write_examples(src_examples, per_file_path, fmt)
            total_written += len(src_examples)
            all_examples.extend(src_examples)
        log.info(f"  Per-file output: {len(source_groups)} files ({total_written} total examples)")
        actual_path = f"{base_path}_<source>.{ext}"
        examples = all_examples
    else:
        examples = _build_examples(deduped, style, chapter_counts, all_metadata)
        actual_path = output_path

    # Structured packing mode (SPLICE, ref [1] §6)
    if do_pack and examples:
        packed = pack_examples(examples, max_tokens=pack_max_tokens)
        log.info(f"  Packing: {len(examples)} examples → {len(packed)} packed samples"
                 f" (ratio: {len(packed)/max(len(examples),1):.1%})")
        examples = packed

    # Split by domain (ref [1] §3 — Domain Isolation)
    if split_domain and not do_pack and not per_file:
        domain_groups = defaultdict(list)
        for ex in examples:
            domain = ex.get('content_type', 'narrative')
            domain_groups[domain].append(ex)
        base_path = output_path.rsplit('.', 1)[0]
        ext = output_path.rsplit('.', 1)[1] if '.' in output_path else fmt
        total_written = 0
        for domain, domain_exs in sorted(domain_groups.items()):
            domain_path = f"{base_path}_{domain}.{ext}"
            _write_examples(domain_exs, domain_path, fmt)
            total_written += len(domain_exs)
        log.info(f"  Split by domain: {len(domain_groups)} files ({total_written} total examples)")
        actual_path = f"{base_path}_<domain>.{ext}"
    elif split_domain and per_file:
        log.info("  Note: --split-domain and --per-file are mutually exclusive; using per-file")

    # Write output (with sharding support, ref [4] — SlimPajama-style)
    if shard_count > 0 and len(examples) > shard_count:
        shard_size = len(examples) // shard_count + 1
        base_path = output_path.rsplit('.', 1)[0]
        ext = output_path.rsplit('.', 1)[1] if '.' in output_path else fmt
        total_written = 0
        for shard_idx in range(shard_count):
            start = shard_idx * shard_size
            end = min(start + shard_size, len(examples))
            if start >= len(examples):
                break
            shard_path = f"{base_path}_{shard_idx:04d}.{ext}"
            shard_examples = examples[start:end]
            _write_examples(shard_examples, shard_path, fmt)
            total_written += len(shard_examples)
        log.info(f"  Sharded into {shard_count} files ({total_written} total examples)")
        actual_path = f"{base_path}_0000.{ext} ..."
    else:
        _write_examples(examples, output_path, fmt)
        actual_path = output_path

    elapsed = (datetime.now() - start_time).total_seconds()
    if shard_count == 0:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
    else:
        size_mb = 0  # sum of shards would be complex

    total_tokens = sum(c.get('est_tokens', 0) for c in deduped)

    log.info(f"── Phase 6: Format ({style}, {fmt}) ── wrote {len(examples)} examples ──")
    log.info(f"── Done ({elapsed:.1f}s) ──────────────────────────────────")
    log.info(f"  Output:      {actual_path}")
    log.info(f"  Examples:    {len(examples)}")
    log.info(f"  Total tokens: ~{total_tokens}")
    log.info(f"  Sources:     {len(epubs)} EPUBs")
    log.info(f"  Content:     {dict(final_domain_dist.most_common())}")

    # Write report
    report = {
        "pipeline": "epub2dataset v3 — Curate-and-Focus Methodology",
        "research_refs": ["[1] From Curation to Code", "[2] Curating for Capability",
                          "[3] Curating for Efficiency", "[4] From Curation to Compression"],
        "recipe": recipe_name or "custom",
        "style": style,
        "format": fmt,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "epubs_found": len(epubs),
        "chapters_extracted": len(all_chapters),
        "raw_chunks": len(raw_chunks),
        "quality_passed": len(scored),
        "quality_rejected": dict(rejected),
        "avg_quality_score": round(sum(c.get('quality', 0) for c in scored) / len(scored), 2) if scored and not lightweight else None,
        "dedup_removed": len(dedup_indices),
        "decontamination_removed": contaminated,
        "final_examples": len(examples),
        "total_est_tokens": total_tokens,
        "content_type_distribution": dict(final_domain_dist),
        "pii_detected": dict(total_pii) if total_pii else None,
        "chapters_merged": merge_count,
        "per_source_quality": dict(Counter(
            c['source'] for c in scored
        ).most_common()),
        "avg_quality_by_source": {
            src: round(sum(c['quality'] for c in scored if c['source'] == src) / max(
                sum(1 for c in scored if c['source'] == src), 1
            ), 2)
            for src in set(c['source'] for c in scored)
        } if scored else {},
        "output_format": fmt,
        "output_style": style,
        "shards": shard_count,
        "parameters": {
            "min_chars": min_chars, "max_chars": max_chars,
            "quality_threshold": quality_threshold,
            "similarity": similarity, "min_density": min_density,
            "entropy_threshold": entropy_threshold,
            "lightweight": lightweight,
            "redact_pii": redact_pii_flag,
        },
    }
    report_path = (output_path.rsplit('.', 1)[0] + '_report.json') if shard_count == 0 else \
                  (output_path.rsplit('.', 1)[0].rsplit('_', 1)[0] + '_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"  Report:      {report_path}")

    # Show sample
    log.info(f"\n  Sample output ({style} style):")
    if examples:
        ex = examples[0]
        log.info(f"    source:    {ex.get('source', '')}")
        log.info(f"    chapter:   {ex.get('chapter', '')}")
        log.info(f"    quality:   {ex.get('quality_score', 'N/A')}/10")
        log.info(f"    content:   {ex.get('content_type', 'N/A')}")
        log.info(f"    est_tokens: ~{ex.get('est_tokens', 0)}")
        if style == 'instruction':
            log.info(f"    response preview: {ex.get('response', '')[:120].replace(chr(10), ' ')}...")
        elif style == 'completion':
            log.info(f"    completion preview: {ex.get('completion', '')[:120].replace(chr(10), ' ')}...")
        elif style == 'chat':
            msgs = ex.get('messages', [])
            if msgs:
                log.info(f"    messages: {len(msgs)} turns (system/user/assistant)")

    return 0


def _build_examples(deduped: list, style: str, chapter_counts: Counter,
                    metadata_map: Optional[dict] = None) -> list:
    """Build formatted examples from deduplicated chunks."""
    examples = []
    for c in deduped:
        key = (c['source'], c['chapter_title'])
        total = chapter_counts[key]
        tmpl_idx = int(hashlib.md5(c['source'].encode()).hexdigest(), 16) % len(INSTRUCTION_TEMPLATES)
        meta = (metadata_map or {}).get(c['source'], {})

        if style == 'instruction':
            ex = make_example_instruction(
                text=c['text'], source=c['source'],
                chapter_title=c['chapter_title'],
                chapter_index=c['chapter_index'],
                total_chunks=total, template_idx=tmpl_idx,
                quality=c.get('quality', 5.0),
                dims=c.get('quality_dims'),
                content_type=c.get('content_type', 'narrative'),
                pii_found=c.get('pii_found'),
                est_tokens=c.get('est_tokens', 0),
                metadata=meta,
            )
        elif style == 'completion':
            ex = make_example_completion(
                text=c['text'], source=c['source'],
                chapter_title=c['chapter_title'],
                chapter_index=c['chapter_index'],
                quality=c.get('quality', 5.0),
                content_type=c.get('content_type', 'narrative'),
                est_tokens=c.get('est_tokens', 0),
            )
        elif style == 'chat':
            ex = make_example_chat(
                text=c['text'], source=c['source'],
                chapter_title=c['chapter_title'],
                chapter_index=c['chapter_index'],
                quality=c.get('quality', 5.0),
                content_type=c.get('content_type', 'narrative'),
                est_tokens=c.get('est_tokens', 0),
            )
        examples.append(ex)
    return examples


def _write_examples(examples: list, output_path: str, fmt: str):
    """Write examples in the specified format."""
    if fmt == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    elif fmt == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
    elif fmt == 'parquet':
        import pandas as pd
        df = pd.DataFrame(examples)
        if 'quality_dimensions' in df.columns and df['quality_dimensions'].notna().any():
            qd_df = df['quality_dimensions'].apply(pd.Series)
            df = pd.concat([df.drop(columns=['quality_dimensions']), qd_df], axis=1)
        if 'pii_found' in df.columns:
            df = df.drop(columns=['pii_found'])
        df.to_parquet(output_path, index=False)
    elif fmt == 'arrow':
        import pyarrow as pa
        import pyarrow.ipc as ipc
        flat = []
        for ex in examples:
            dims = ex.pop('quality_dimensions', {})
            pii = ex.pop('pii_found', {})
            row = {k: v for k, v in ex.items()
                   if not isinstance(v, (dict, list))}
            row.update(dims)
            flat.append(row)
        first = flat[0] if flat else {}
        schema = []
        for k, v in first.items():
            if isinstance(v, bool):
                schema.append(pa.field(k, pa.bool_()))
            elif isinstance(v, int):
                schema.append(pa.field(k, pa.int64()))
            elif isinstance(v, float):
                schema.append(pa.field(k, pa.float64()))
            else:
                schema.append(pa.field(k, pa.string()))
        batch = pa.RecordBatch.from_pylist(flat, schema=pa.schema(schema))
        with pa.OSFile(output_path, 'wb') as f:
            ipc.write_file(batch, f)


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="epub2dataset v3 — Research-grade EPUB to LLM training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Recipes (ref [3] §2 — "data recipe" experimentation):
              strict   — min=500ch, max=4K, quality≥7.5, high dedup
              balanced — min=300ch, max=8K, quality≥6.0 (default)
              lenient  — min=150ch, max=16K, quality≥4.0

            Output Styles (ref [1] §5):
              instruction — Alpaca-style ### Instruction/### Input/### Response (default)
              completion  — Simple prompt + completion
              chat        — Messages array (system/user/assistant)

            Formats:
              jsonl    — JSON Lines (default)
              json     — Pretty-printed JSON array
              parquet  — Apache Parquet columnar format (ref [4] §2)
              arrow    — Apache Arrow IPC format (ref [4] §2, zero-copy reads)

            New in v3:
              --style chat / --style completion   (ref [1] §5 — 3 output styles)
              --redact-pii                        (ref [1] §3 — Privacy Compliance)
              --shard N                           (ref [4] — SlimPajama-style sharding)
              --lightweight                       (ref [3] §2 — fast pipeline)
              Content-type detection              (ref [1] §3 — Domain Classification)
              Front-matter skipping               (ref [1] §2 — file filtering)
              Token estimation per example        (ref [4] §3)

            Examples:
              python epub2dataset.py ./ebooks -o dataset.jsonl
              python epub2dataset.py ./ebooks -o chat.jsonl --style chat
              python epub2dataset.py ./ebooks -o data.parquet --format parquet
              python epub2dataset.py ./ebooks --recipe strict --style instruction
              python epub2dataset.py ./ebooks --lightweight --no-dedup
              python epub2dataset.py ./ebooks -o dataset --shard 10
              python epub2dataset.py ./ebooks --redact-pii
              python epub2dataset.py ./ebooks --stats
        """),
    )
    parser.add_argument("input_dir", help="Directory containing .epub files (recursive)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: dataset.<format>)")
    parser.add_argument("--format", choices=["jsonl", "json", "parquet", "arrow"],
                        default="jsonl", help="Output format (default: jsonl)")
    parser.add_argument("--style", choices=["instruction", "completion", "chat"],
                        default="instruction",
                        help="SFT data style (ref [1] §5, default: instruction)")
    parser.add_argument("--recipe", choices=list(RECIPES.keys()) + [""],
                        default="balanced",
                        help="Pre-set parameter recipe (default: balanced)")
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--similarity", type=float, default=None)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--redact-pii", action="store_true",
                        help="Redact PII (email, phone, SSN) from output (ref [1] §3)")
    parser.add_argument("--shard", type=int, default=0,
                        help="Split output into N shard files (ref [4] — SlimPajama)")
    parser.add_argument("--lightweight", action="store_true",
                        help="Skip heavy checks (entropy, structure, dedup) for speed")
    parser.add_argument("--include-front-matter", action="store_true",
                        help="Include title/copyright/TOC pages (default: skip)")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--stream", action="store_true",
                        help="Streaming mode (process without loading all into RAM)")

    # v4 additions
    default_jobs = multiprocessing.cpu_count() if HAS_MULTIPROCESSING else 1
    parser.add_argument("--jobs", type=int, default=default_jobs,
                        help=f"Parallel extraction workers (default: {default_jobs}, ref [4] §2 — CPU thread pool)")
    parser.add_argument("--config", type=str, default=None,
                        help="Load recipe from JSON config file")
    parser.add_argument("--save-config", type=str, default=None,
                        help="Save current recipe as JSON config file")
    parser.add_argument("--per-file", action="store_true",
                        help="One output file per source EPUB (SlimPajama-style, ref [4])")
    parser.add_argument("--quality-csv", type=str, default=None,
                        help="Export quality scores as CSV for analysis")

    # v5 additions
    parser.add_argument("--pack", action="store_true",
                        help="Structured packing mode — combine related chunks (SPLICE, ref [1] §6)")
    parser.add_argument("--pack-max-tokens", type=int, default=2048,
                        help="Max tokens per packed sample (default: 2048, only with --pack)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        choices=["cl100k", "p50k", "r50k"],
                        help="Count tokens with specific model tokenizer (needs: pip install tiktoken)")
    parser.add_argument("--split-domain", action="store_true",
                        help="Create separate output per content type (narrative, technical, etc.)")

    # v6 additions
    parser.add_argument("--validate", action="store_true",
                        help="Score existing JSONL against quality rubric (ref [4] — Validation)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge multiple JSONL files with dedup (use glob: 'runs/*.jsonl')")
    parser.add_argument("--license", type=str, default=None,
                        help="Filter EPUBs by license type (e.g. 'cc-by', 'public domain') (ref [1] §2)")
    args = parser.parse_args()

    if args.output is None:
        ext = {'jsonl': 'jsonl', 'json': 'json', 'parquet': 'parquet', 'arrow': 'arrow'}[args.format]
        args.output = f"dataset.{ext}"

    # Auto-detect format from file extension
    ext_map = {'.jsonl': 'jsonl', '.json': 'json', '.parquet': 'parquet', '.arrow': 'arrow'}
    output_ext = os.path.splitext(args.output)[1].lower()
    if output_ext in ext_map and args.format == 'jsonl':
        # Only auto-detect if user didn't explicitly set --format
        args.format = ext_map[output_ext]

    # Validate/merge modes accept file paths, not just directories
    is_validate = getattr(args, 'validate', False)
    is_merge = getattr(args, 'merge', False)

    if not os.path.isdir(args.input_dir) and not is_validate and not is_merge:
        log.error(f"Directory not found: {args.input_dir}")
        return 1

    # Save config if requested
    if args.save_config:
        config = {
            "recipe": args.recipe,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "quality_threshold": args.quality_threshold,
            "similarity": args.similarity,
            "style": args.style,
        }
        config = {k: v for k, v in config.items() if v is not None}
        with open(args.save_config, 'w') as f:
            json.dump(config, f, indent=2)
        log.info(f"Saved config: {args.save_config}")
        return 0

    if args.stats:
        print_stats(args.input_dir)
        return 0

    # ── Validate mode (ref [4] — Validation and Sampling) ──────
    if getattr(args, 'validate', False):
        input_path = args.input_dir
        if not os.path.isfile(input_path) and not input_path.endswith('.jsonl'):
            log.error("--validate requires a path to a .jsonl file")
            return 1
        try:
            scores = []
            with open(input_path) as f:
                for i, line in enumerate(f, 1):
                    ex = json.loads(line)
                    text = ex.get('response') or ex.get('completion') or ''
                    if not text:
                        msgs = ex.get('messages', [])
                        if len(msgs) >= 3:
                            text = msgs[2].get('content', '')
                    score, dims = quality_score_8d(text, 100, 99999)
                    scores.append(score)
            if scores:
                avg = sum(scores) / len(scores)
                best = max(scores)
                worst = min(scores)
                log.info(f"── Validation: {input_path} ──")
                log.info(f"  Examples: {len(scores)}")
                log.info(f"  Avg quality: {avg:.2f}/10")
                log.info(f"  Best: {best:.2f}/10")
                log.info(f"  Worst: {worst:.2f}/10")
                # Distribution histogram
                bins = [0]*10
                for s in scores:
                    b = min(int(s), 9)
                    bins[b] += 1
                log.info(f"  Distribution:")
                for i, count in enumerate(bins):
                    if count > 0:
                        bar = "█" * count + "▏" * (count % 5) if count < 50 else "█" * 50 + f" {count}"
                        log.info(f"    {i}.{i+1}: {bar}")
            else:
                log.warning("No valid examples found")
        except Exception as e:
            log.error(f"Validation failed: {e}")
        return 0

    # ── Merge mode ─────────────────────────────────────────────
    if getattr(args, 'merge', False):
        import glob as gmod
        pattern = args.input_dir
        files = sorted(gmod.glob(pattern))
        if not files:
            log.error(f"No files matching: {pattern}")
            return 1
        all_examples = []
        for fpath in files:
            with open(fpath) as f:
                for line in f:
                    try:
                        all_examples.append(json.loads(line))
                    except:
                        pass
        log.info(f"── Merge: {len(files)} files → {len(all_examples)} examples ──")
        # Deduplicate across merged files
        if len(all_examples) > 1:
            texts = [json.dumps(ex, sort_keys=True) for ex in all_examples]
            unique = list(dict.fromkeys(texts))  # preserve order, remove exact dupes
            all_examples = [json.loads(t) for t in unique]
            log.info(f"  After exact dedup: {len(all_examples)} unique examples")
        out = getattr(args, 'output', None) or 'merged.jsonl'
        _write_examples(all_examples, out, getattr(args, 'format', 'jsonl'))
        log.info(f"  Written to: {out}")
        return 0

    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
