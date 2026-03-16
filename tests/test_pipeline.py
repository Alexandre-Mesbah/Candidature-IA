from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from study_genie_extract.chunking import chunk_text
from study_genie_extract.pipeline import build_index, prepare_documents
from study_genie_extract.retrieval import cosine_similarity, retrieve_top_k
from study_genie_extract.types import Document


VOCABULARY = ["derivative", "cubed", "gradient", "photosynthesis", "sigmoid"]


def fake_embedding(text: str) -> list[float]:
    lowered = text.lower()
    return [1.0 if token in lowered else 0.0 for token in VOCABULARY]


def fake_embed_batch(texts: list[str], model: str, api_key: str | None) -> list[list[float]]:
    return [fake_embedding(text) for text in texts]


def fake_embed_query(text: str, model: str, api_key: str | None) -> list[float]:
    return fake_embedding(text)


class PipelineTest(unittest.TestCase):
    def test_prepare_documents_normalizes_newlines(self) -> None:
        documents = [Document(document_id="doc_1", title="Notes", text="Line 1\r\n\r\nLine 2\rLine 3")]
        prepared = prepare_documents(documents)
        self.assertEqual(prepared[0].text, "Line 1\n\nLine 2\nLine 3")

    def test_chunk_text_returns_ordered_slices(self) -> None:
        text = "A" * 90 + "\n\n" + "B" * 90 + "\n\n" + "C" * 90
        chunks = chunk_text(text, max_chunk_size=120, overlap=20, separator="\n\n")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertTrue(all(chunk.content for chunk in chunks))

    def test_cosine_similarity_is_one_for_identical_vectors(self) -> None:
        vector = [1.0, 0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(vector, vector), 1.0, places=12)

    def test_top_k_retrieval_returns_expected_chunk(self) -> None:
        documents = [
            Document(
                document_id="calc_1",
                title="Calculus Notes",
                text="The derivative of x cubed is three x squared.",
            ),
            Document(
                document_id="bio_1",
                title="Biology Notes",
                text="Photosynthesis turns light into chemical energy.",
            ),
        ]
        index = build_index(documents, model="fake-model", api_key=None, embed_chunks=fake_embed_batch)
        hits = retrieve_top_k(
            "rule for x cubed derivative",
            index,
            top_k=1,
            model="fake-model",
            api_key=None,
            embed_query=fake_embed_query,
        )

        self.assertEqual(hits[0].chunk.document_id, "calc_1")


if __name__ == "__main__":
    unittest.main()
