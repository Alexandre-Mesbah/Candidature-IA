from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


def create_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
) -> list[float]:
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    embeddings = create_embeddings([text], model=model, api_key=api_key)
    return embeddings[0]


def create_embeddings(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    valid_inputs = [
        {"index": index, "text": text.strip()}
        for index, text in enumerate(texts)
        if text.strip()
    ]
    if not valid_inputs:
        raise ValueError("no valid texts to embed")

    payload = {
        "model": model,
        "input": [item["text"] for item in valid_inputs],
    }
    response_json = post_embeddings_request(payload, api_key=resolve_api_key(api_key))

    response_embeddings: list[list[float]] = [[] for _ in texts]
    for item in response_json["data"]:
        original_index = valid_inputs[item["index"]]["index"]
        response_embeddings[original_index] = [float(value) for value in item["embedding"]]

    missing_vectors = [index for index, vector in enumerate(response_embeddings) if not vector]
    if missing_vectors:
        raise RuntimeError(f"missing embeddings for indexes: {missing_vectors}")

    return response_embeddings


def resolve_api_key(explicit_api_key: str | None) -> str:
    api_key = explicit_api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate embeddings")
    return api_key


def post_embeddings_request(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = Request(
        EMBEDDINGS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"embedding request failed with status {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"embedding request failed: {error.reason}") from error
