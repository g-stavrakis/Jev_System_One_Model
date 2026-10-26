# Jev Walkthrough

Notebook examples for using TypeSafe's Jev (`jev-latest`) for classification, routing, groundedness checks, and relevance scoring, plus a comparison with GPT models using structured output.

## Contents

- [jev_usecases.ipynb](jev_usecases.ipynb): introductory examples covering agent/model routing, verification, scoring, and token usage.
- [model_comparison.ipynb](model_comparison.ipynb): benchmarks on product reviews, RAG groundedness, banking intents, and support tickets, measuring classification quality, latency, and estimated cost.
- `utils/`: model calls, evaluation metrics, schema/prompt helpers, and plotting utilities.
- `final_results.csv` and `charts/`: saved benchmark results and comparison plots.

## Getting started

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/), and a TypeSafe API key. The GPT comparisons also require an OpenAI API key.

```bash
uv sync
uv run --with jupyter jupyter lab
```

Create a `.env` file in the repository root:

```dotenv
TYPESAFE_API_KEY=your_typesafe_key
OPENAI_API_KEY=your_openai_key
```

Open a notebook and run this cell **before any SDK or `utils.model_functions` imports** to load your credentials:

```python
from dotenv import load_dotenv
load_dotenv()
```

Run the remaining cells in order, starting with `jev_usecases.ipynb` for the basic examples. Launch Jupyter from the repository root so local imports and output paths resolve correctly.

The comparison notebook downloads datasets and makes billable API calls. It writes to `final_results.csv` and saves plots under `charts/`. Cost estimates use hardcoded rates in the code; review those rates and the configured model names before rerunning benchmarks.
