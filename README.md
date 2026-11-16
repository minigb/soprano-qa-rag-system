# Soprano QA RAG System

Measure-aware question answering for five soprano works. A hybrid BM25 +
embedding retriever selects the relevant sources, and one model call writes the
answer from them. Sumi Jo's direct feedback ranks first, then the reviewed
expert annotations, then web coaching notes. Sumi Jo's feedback currently covers
Nella fantasia. English questions are translated into Korean for retrieval and
answered in English.

This repository is the RAG/LLM layer, a Python library with a CLI. It reads the
sibling [soprano-qa-dataset](https://github.com/minigb/soprano-qa-dataset)
repository, and the sibling [soprano-qa-demo](https://github.com/minigb/soprano-qa-demo)
web application uses it.

## Answer model

The answer model is the only setting that differs between hosts.

| `llm.provider` | Answer model | Used by |
| --- | --- | --- |
| `local`, the default | Qwen3-14B-Q6_K through llama-cpp-python | local runs and the demo application |
| `vertex_gemini` | `gemini-3.8-flash` on Vertex AI | the soprano-lab web demo, which has no GPU |

Set it in `config/settings.json` or with the `SOPRANO_QA_LLM_PROVIDER`
environment variable, which overrides the file. Retrieval, prompts and source
priority are the same for both. [docs/gemini-vertex.md](docs/gemini-vertex.md)
describes the Gemini setup.

## Setup

### 1. Clone the three repositories side by side

```bash
mkdir soprano-qa-workspace && cd soprano-qa-workspace
git clone https://github.com/minigb/soprano-qa-dataset.git
git clone https://github.com/minigb/soprano-qa-rag-system.git
git clone https://github.com/minigb/soprano-qa-demo.git
```

The default paths in `config/settings.json` expect these directory names. For
another layout, set `SOPRANO_QA_RAG_DATASET_ROOT`, `SOPRANO_QA_MODEL_PATH` or
`SOPRANO_QA_EMBEDDING_MODEL_PATH`. The demo repository is needed only for the
web application.

### 2. Create the shared Conda environment

```bash
cd soprano-qa-dataset
conda env create -f environment.yml
conda activate soprano-qa
cd ../soprano-qa-rag-system
python -m pip install -r requirements.txt
```

For NVIDIA CUDA 12.4, add
`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124` to the
`pip install` command.

### 3. Build the corpus and download the models

```bash
python scripts/build_corpus.py
python scripts/download_embedding_model.py
python scripts/download_model.py
```

| Checkpoint | Size | Needed for |
| --- | --- | --- |
| `Qwen3-14B-Q6_K.gguf` | 12.1 GB | generating answers |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | 639 MB | hybrid retrieval |

Both models use all available GPU layers by default. If they must share a
smaller GPU, lower `llm.n_gpu_layers` and `retrieval.n_gpu_layers` in
`config/settings.json`.

### 4. Check the installation

```bash
python scripts/ask.py --piece die-forelle --measures 2-5 \
  --question "피아노 반주에서 두 번째 박의 악센트는 무엇을 나타내는가?" --no-generate
python -m unittest discover
```

`--no-generate` runs retrieval only, without loading the answer model.

## Usage

```bash
python scripts/ask.py --piece nella-fantasia \
  --question "이 곡은 어떤 발성으로 불러야 하나요?"
```

- `--measures 28-30` selects a score range. Omit it for a question about the
  whole piece.
- `--language en` answers in English.
- `--json` prints the evidence and metadata along with the answer.

From Python:

```python
from soprano_qa.service import ask

result = ask(
    piece_id="nella-fantasia",
    question="이 곡은 어떤 발성으로 불러야 하나요?",
    measure_range=None,  # or a (start, end) tuple
    generate=True,
    language="ko",       # or "en"
)
print(result["answer"])
```

The piece IDs are `die-forelle`, `in-flowery-clouds`, `la-capinera`,
`nella-fantasia` and `una-voce-poco-fa`.

## Documentation

- [docs/rag-llm-pipeline-method.md](docs/rag-llm-pipeline-method.md): how an
  answer is produced, step by step
- [docs/source-priority.md](docs/source-priority.md): how Sumi Jo's feedback is
  ranked first
- [docs/rag-security-research.md](docs/rag-security-research.md): the
  prompt-injection screens and their limits
- [docs/pipeline-evolution-log.md](docs/pipeline-evolution-log.md): the history
  of pipeline versions. Until 2026-11-16 the default made several model calls
  per question; commit `bbb84da` is that pipeline's last version.
- [evaluation/README.md](evaluation/README.md): evaluation runs and reports
