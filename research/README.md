# Research Papers — epub2dataset Foundation

This directory contains the four research papers that define the **Curate-and-Focus Methodology** implemented by epub2dataset.

## Papers

| # | Title | Pages | Focus |
|---|-------|-------|-------|
| 1 | **From Curation to Code: A Practical Framework for Building Single-Domain Datasets for Low-CPU LLM Training** | 23 | Core pipeline: multi-stage curation, quality scoring (Code2Doc 8-dim rubric), SFT formatting, PEFT recommendations |
| 2 | **Curating for Capability: A Low-CPU Framework for Building Specialized LLM Datasets** | 20 | Domain-specific data acquisition, SLM selection (Phi-3, TinyLlama), streaming/preprocessing optimization |
| 3 | **Curating for Efficiency: A Blueprint for Building Specialized LLMs on Low-Power Hardware** | 19 | Data recipe concept, two-pronged curation strategy, token efficiency, source provenance |
| 4 | **From Curation to Compression: Building High-Efficiency LLM Datasets for Resource-Constrained Training** | 26 | Data formats (Arrow, Parquet, TFRecord vs JSONL), Unicode NFKC normalization, I/O optimization |

## How They Map to epub2dataset

| epub2dataset Feature | Source Paper(s) |
|---------------------|-----------------|
| 6-phase pipeline | Paper [1] §3 — Multi-Stage Data Curation Pipeline |
| 8-dimension quality scoring | Paper [1] §3 — Code2Doc rubric (≥ 6.0/10 threshold) |
| Instruction / Completion / Chat styles | Paper [1] §5 — Three SFT formatting styles |
| Parquet & Arrow output | Paper [4] §2 — Columnar binary formats for low-CPU I/O |
| JSONL default | Paper [1] §5 — Industry standard for fine-tuning |
| MinHash LSH dedup | Paper [1] §3, Paper [3] §2 — Global vs local deduplication |
| Entropy-based filtering | Paper [1] §6, Paper [3] §2 — Lightweight data selection |
| PII detection | Paper [1] §3 — Privacy Compliance (Microsoft Presidio) |
| Front-matter skipping | Paper [1] §2 — File extension/size/license filtering |
| NFKC normalization | Paper [4] §3 — Unicode normalization |
| Content-type classification | Paper [1] §3 — Domain Isolation & Classification |
| Output sharding | Paper [4] — SlimPajama's ~59,000 JSONL files |
| Three recipes (strict/balanced/lenient) | Paper [3] §2 — "Data recipe" experimentation |
| Lightweight mode | Paper [3] §2 — Two-pronged curation strategy |
| Token estimation | Paper [4] §3 — Tokenization as CPU bottleneck |

## Citation

If you use this research or the epub2dataset tool in your work, please cite the original papers (available in this directory).
