# Fact-Checking System Using the LIAR Dataset

Deployed on Streamlit: [Click HERE](https://unshdee-factcheckliar-app-2y5wr1.streamlit.app/)

## Model Overview

This model is a fact-checking system built on the LIAR dataset. It combines both sparse (BM25) and dense (FAISS) retrieval techniques to locate relevant claims and uses a fine-tuned BERT-based classifier to predict the veracity of user-provided statements. The system can generate responses via a local LLM (Ollama) or fall back to template-based generation.

## Features

- **Index Persistence**: FAISS and BM25 indexes are cached to disk, reducing startup time from ~30-60s to ~2-3s on subsequent runs
- **LLM Integration**: Optional Ollama integration for more natural, context-aware responses
- **Hybrid Retrieval**: Combines BM25 (sparse) and FAISS (dense) retrieval for better claim matching
- **Template Fallback**: Works without LLM using structured template responses

## Training Data

- Dataset: LIAR dataset, which contains thousands of labeled political statements along with metadata such as speaker information, job titles, and context.
- Labels: The system classifies statements into six categories: `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, and `true`.

## Model Details

### Retrieval Component:

- Sparse Retrieval: BM25 index over claim statements.
- Dense Retrieval: FAISS index built on embeddings from the "all-MiniLM-L6-v2" SentenceTransformer.
- Score Fusion: A weighted sum of BM25 and dense retrieval scores is used to identify the most relevant similar claim.

### Classification Component:

- Model: [unshDee/liar_qa](https://huggingface.co/unshDee/liar_qa) - BERT (bert-base-uncased) fine-tuned on the LIAR dataset for six-class veracity prediction.
- Output: The predicted label (e.g., "false") is used in the final fact-checking response.

### Response Generation:

- **With LLM**: Uses Ollama to generate contextual, natural language responses
- **Without LLM**: Falls back to a template-based system that formats the output

## Installation

### Install dependencies (using uv)

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

### Classification Model

On the first run, the model will be downloaded (to a folder named `classifier_model`) from the Hugging Face model hub.

[unshDee/liar_qa](https://huggingface.co/unshDee/liar_qa)

### Ollama Setup (Optional)

To use LLM-based response generation, install and run Ollama:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull gemma3:4b-it-qat`
3. Start the server: `ollama serve`

Configure in `.env`:
```
LLM_PROVIDER=ollama
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b-it-qat
```

## Usage

### macOS Apple Silicon (M1/M2/M3/M4) Users

If you encounter segmentation faults when running on macOS with Apple Silicon, set these environment variables:

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
```

Or prefix your commands:

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 uv run python main.py --query "your claim"
```

You can add these to your shell profile (`~/.zshrc` or `~/.bashrc`) for convenience:

```bash
# Add to ~/.zshrc
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
```

### Command Line Arguments

```
--query         Claim to fact-check
--verbose       Show detailed evidence and context
--no-llm        Disable LLM generation (use template only)
--train_classifier  Train the classifier from scratch
```

### via Terminal

```bash
# Basic usage (with LLM if available)
uv run python main.py --query "Is it true that Barack Obama was born in Kenya?"

# Verbose mode with detailed evidence
uv run python main.py --query "Is climate change a hoax?" --verbose

# Without LLM (template-based response)
uv run python main.py --query "Is climate change a hoax?" --no-llm

# More examples
uv run python main.py --query "Is it true that the COVID-19 vaccine contains microchips?"
uv run python main.py --query "Is it true that 5G networks cause severe health issues?"
uv run python main.py --query "Is it true that illegal immigrants are the primary cause of crime in the United States?"
```

### via Streamlit

```bash
uv run streamlit run app.py
```

The Streamlit UI provides:
- Text input for claims
- "Show detailed evidence" checkbox (verbose mode)
- "Use LLM for response generation" checkbox

## Cache Directory

On first run, indexes are built and cached to the `cache/` directory:

```
cache/
  ├── faiss_index.bin      # FAISS dense embeddings index (~16MB)
  ├── bm25_index.pkl       # BM25 sparse index (~2MB)
  └── dataset_hash.txt     # MD5 hash for cache invalidation
```

The cache is automatically invalidated if the dataset (`data/train.tsv`) changes.

## Performance

| Metric | Before | After |
|--------|--------|-------|
| First run startup | ~30-60s | ~30-60s (builds + caches) |
| Subsequent startup | ~30-60s | **~2-3s** (loads from cache) |
| Cache size | N/A | ~18 MB |

---

## To evaluate the performance of the fact-checking system:

- **Assess the Classifier**: Use accuracy, precision, recall, and F1-score on a held-out test set from the LIAR dataset to measure how well the model distinguishes between the six labels.
- **Evaluate Retrieval Effectiveness**: Compute retrieval metrics like recall@k and mean reciprocal rank (MRR) to ensure that the BM25+FAISS hybrid returns the most relevant supporting claims.
- **User Testing**: Perform usability studies with real users to determine if the final natural language responses are clear, informative, and useful in verifying claims.
