# Soprano QA RAG System

Local, measure-aware retrieval and Qwen-based answer generation for five
soprano works. This repository is the RAG/LLM layer and uses the sibling
dataset and demo repositories described below.

## Setup

### 1. Clone the three repositories

Keep the default directory names as immediate siblings:

```bash
mkdir soprano-qa-workspace
cd soprano-qa-workspace
git clone https://github.com/minigb/soprano-qa-dataset.git
git clone https://github.com/minigb/soprano-qa-rag-system.git
git clone https://github.com/minigb/soprano-qa-demo.git
```

The resulting layout must be:

```text
soprano-qa-workspace/
├── soprano-qa-dataset/
├── soprano-qa-rag-system/
└── soprano-qa-demo/
```

### 2. Create the shared Conda environment

Install Conda (Miniconda or Miniforge) and initialize it for your shell first.
All three repositories then use the single `soprano-qa` environment. Run these
commands from `soprano-qa-workspace/`:

```bash
cd soprano-qa-dataset
conda env create -f environment.yml
conda activate soprano-qa

cd ../soprano-qa-rag-system
python -m pip install -r requirements.txt
```

If `soprano-qa` already exists, run this alternative block from
`soprano-qa-workspace/` to update it without pruning packages:

```bash
cd soprano-qa-dataset
conda env update -n soprano-qa -f environment.yml
conda activate soprano-qa
cd ../soprano-qa-rag-system
python -m pip install -r requirements.txt
```

For NVIDIA CUDA 12.4, replace the `pip install` command in the chosen block
with:

```bash
python -m pip install -r requirements.txt \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Other platforms may build `llama-cpp-python` locally or use an appropriate
platform-specific wheel.

### 3. Build the corpus and optionally download the model

```bash
python scripts/build_corpus.py
python scripts/download_model.py  # Optional; needed for generated answers
```

The default Qwen GGUF checkpoint needs approximately 5 GB. Retrieval-only
queries work without the model by passing `--no-generate`.

### 4. Verify the installation

```bash
python scripts/ask.py \
  --piece die-forelle \
  --measures 2-5 \
  --question "피아노 반주에서 두 번째 박의 악센트는 무엇을 나타내는가?" \
  --no-generate

python -m unittest discover -v
```

## Usage

Run a generated, measure-specific answer:

```bash
python scripts/ask.py \
  --piece die-forelle \
  --measures 28-30 \
  --question "28마디부터 분위기 변화를 어떻게 표현해야 하나요?"
```

Omit `--measures` for a whole-work question. Add `--json` for structured
retrieval, evidence, rights, citation, and answer metadata.

Applications can import the service facade while this repository is on
`PYTHONPATH`:

```python
from soprano_qa.service import ask

result = ask(
    piece_id="die-forelle",
    question="피아노 반주에서 두 번째 박의 악센트는 무엇을 나타내는가?",
    measure_range=(2, 5),
    generate=True,
)
```

## Configuration

Defaults are repository-relative and assume the sibling layout above.

| Setting | Purpose |
| --- | --- |
| `SOPRANO_QA_RAG_DATASET_ROOT` | Override the dataset/corpus repository |
| `SOPRANO_QA_MODEL_PATH` | Override the local GGUF checkpoint |
| `config/settings.json` | Corpus, model, retrieval, and generation defaults |
| `config/feature_overrides.json` | Optional recurring measure-range overrides |

## Project notes

The corpus combines expert measure annotations with answer-eligible web
database records. Measure-scoped questions prioritize overlapping expert
evidence; unsupported questions can use a clearly marked internal-knowledge
fallback without corpus citations.

See [docs/measure-aware-rag.md](docs/measure-aware-rag.md) for the detailed
retrieval, provenance, rights, range-routing, and citation design.

To run the web application, continue with the
[demo setup](https://github.com/minigb/soprano-qa-demo#setup).
