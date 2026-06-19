# epub2dataset

**Convert EPUB ebooks into high-quality LLM training datasets — following the Curate-and-Focus Methodology.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

A research-backed tool that transforms `.epub` ebooks into quality-filtered, deduplicated, instruction-style datasets ready for LLM fine-tuning (LoRA/QLoRA). Implements the full 6-phase pipeline from four published research papers on low-CPU dataset construction.

## Features

- **3 Output Styles** — Instruction (Alpaca-style `###` markers), Completion, Chat (messages array)
- **4 Output Formats** — JSONL, JSON, Apache Parquet, Apache Arrow (zero-copy)
- **8-Dimension Quality Scoring** — Code2Doc-inspired rubric: completeness, clarity, coherence, density, structure, length, formatting, uniqueness
- **Semantic Chunking** — Paragraph-boundary splitting with overlap for context continuity
- **Global Deduplication** — MinHash + LSH near-duplicate detection across all sources
- **Domain Classification** — Heuristic content-type detection (narrative, technical, code, dialogue, reference)
- **PII Detection** — Regex-based scanning for email, phone, SSN, IP, credit cards (optional redaction)
- **Front-Matter Skipping** — Auto-detects and skips title pages, copyright, TOC, indices
- **Output Sharding** — SlimPajama-style split into N files
- **Decontamination** — SHA256 fingerprint matching against eval benchmarks
- **3 Pre-set Recipes** — strict (`≥7.5`), balanced (`≥6.0`), lenient (`≥4.0`)
- **Lightweight Mode** — Skips heavy checks for maximum throughput
- **Provenance Tracking** — File hash + source + chapter traceability per example

## Quick Start

```bash
# Install dependencies
pip install ebooklib beautifulsoup4

# Basic conversion
epub2dataset ./ebooks -o dataset.jsonl

# Chat-style for conversational models
epub2dataset ./ebooks -o chat.jsonl --style chat

# Parquet for faster CPU I/O
epub2dataset ./ebooks -o dataset.parquet --format parquet

# Strict quality filter
epub2dataset ./ebooks --recipe strict

# Preview before building
epub2dataset ./ebooks --stats
```

## Recipes

| Recipe | Min Chars | Max Chars | Quality ≥ | Similarity | Use Case |
|--------|-----------|-----------|-----------|------------|----------|
| `strict` | 500 | 4096 | 7.5 | 0.80 | High-quality fine-tuning |
| `balanced` | 300 | 8192 | 6.0 | 0.85 | General purpose (default) |
| `lenient` | 150 | 16384 | 4.0 | 0.90 | Maximum data retention |

## Output Styles

| Style | Schema | Use Case |
|-------|--------|----------|
| `instruction` | `instruction` + `input` + `response` (with `###` markers) | Domain adaptation, SFT |
| `completion` | `prompt` + `completion` | Narrow task fine-tuning |
| `chat` | `messages[]` (system/user/assistant) | Conversational agents |

## Output Formats

| Format | Description | Requires |
|--------|-------------|----------|
| `jsonl` | JSON Lines (default) | — |
| `json` | Pretty-printed JSON array | — |
| `parquet` | Apache Parquet columnar | `pyarrow`, `pandas` |
| `arrow` | Apache Arrow IPC (zero-copy) | `pyarrow` |

## Research Foundation

This tool implements the **Curate-and-Focus Methodology** from four research papers:

1. **"From Curation to Code: A Practical Framework for Building Single-Domain Datasets for Low-CPU LLM Training"**
2. **"Curating for Capability: A Low-CPU Framework for Building Specialized LLM Datasets"**
3. **"Curating for Efficiency: A Blueprint for Building Specialized LLMs on Low-Power Hardware"**
4. **"From Curation to Compression: Building High-Efficiency LLM Datasets for Resource-Constrained Training"**

## Pipeline

```
Phase 1:  Scan & Acquire          →  Recursive EPUB discovery, dedup, front-matter skip
Phase 2:  Extract & Clean         →  Per-chapter extraction, NFKC normalization, PII detection
Phase 3:  Domain Classification   →  Heuristic content-type detection
Phase 4:  Semantic Chunking       →  Paragraph-boundary splitting with overlap
Phase 5:  Quality Scoring + Dedup →  8-dim Code2Doc rubric, MinHash LSH, decontamination
Phase 6:  Format & Write          →  3 styles × 4 formats, with sharding
```

## License

MIT
