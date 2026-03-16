from __future__ import annotations


def preprocess_document_text(text: str) -> str:
    if not text or not text.strip():
        return ""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
