# Study Genie Retrieval Extract

Small public extract of a retrieval component built for Study Genie.

This is not the full application. It only keeps the part needed to show the retrieval pipeline in a simple standalone form.

## Included here

- text preprocessing
- paragraph chunking with overlap
- embedding generation
- cosine-similarity retrieval
- top-k results

The product-specific code was removed.

## Repository layout

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
