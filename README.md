# Study Genie Semantic Retrieval Public Extract

This repository is an adapted standalone extract of one proprietary ML component originally used inside Study Genie.

It is not the full product and it is not a production repository. The goal here is narrower: show the technical logic of a semantic retrieval component in a public, readable form for academic review.

## What this extract keeps

- document preprocessing
- paragraph-based chunking with overlap
- batch embedding generation
- cosine similarity scoring
- top-k retrieval

The chunking logic and the retrieval scoring follow the same core ideas used in the original internal component. The surrounding product code was removed.

## Clean repository structure

```text
Candidature-IA/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   └── sample_documents.jsonl
├── docs/
│   └── error_analysis.md
├── notebooks/
│   └── retrieval_demo.ipynb
├── scripts/
│   └── run_demo.py
├── src/
│   └── study_genie_extract/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── retrieval.py
│       ├── pipeline.py
│       └── types.py
└── tests/
    └── test_pipeline.py
```

## How the pipeline works

1. `preprocessing.py` normalizes raw document text.
2. `chunking.py` splits documents into overlapping chunks.
3. `embeddings.py` sends the chunks to the OpenAI embeddings API in batch.
4. `retrieval.py` embeds the query, computes cosine similarity, and returns the top-k chunks.
5. `pipeline.py` connects the whole flow without any product-specific code.

## Exact run commands

```bash
cd /Users/alex/Desktop/Candidature-IA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key_here"
python scripts/run_demo.py --query "rule for x cubed derivative" --top-k 3
```

## Local verification

```bash
cd /Users/alex/Desktop/Candidature-IA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Notebook and error analysis

- Notebook: [notebooks/retrieval_demo.ipynb](notebooks/retrieval_demo.ipynb)
- Error analysis note: [docs/error_analysis.md](docs/error_analysis.md)

The notebook stays close to the current repository. It loads the demo corpus, shows preprocessing and chunking, then runs retrieval through a deterministic mock embedding path so the example stays reproducible without requiring a live API call.

## Sample input format

The demo reads JSONL documents with these fields:

```json
{"id":"calc_1","title":"Calculus Notes","text":"..."}
```

## What was removed for confidentiality

- authentication, user sessions, and account logic
- database models and persistence
- billing, usage tracking, and observability
- admin routes and internal tooling
- workspace, chat, subscription, and product UI code
- deployment configuration and production infrastructure
- private documents, customer data, and internal prompts

## Public scope

This repository only supports a local standalone demonstration of the retrieval component. It should not be read as a claim about the full Study Genie production stack.
