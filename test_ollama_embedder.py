import json
import math
from unittest import mock

from embedders.model.embed_request import EmbedRequest
from embedders.providers.ollama_embedder import OllamaEmbedder


class _FakeResponse:
    def __init__(self, payload: dict):
        self._bytes = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _mock_urlopen(payloads):
    sequence = iter(payloads)

    def _urlopen(request, timeout):
        try:
            payload = next(sequence)
        except StopIteration as exc:
            raise AssertionError("No more mocked responses available for urlopen") from exc
        return _FakeResponse(payload)

    return _urlopen


def test_ollama_embedder_returns_normalized_embedding():
    embedder = OllamaEmbedder(model_name="bge-m3", normalize=True)
    payloads = [{"embedding": [0.0, 3.0, 4.0]}]
    with mock.patch(
        "embedders.providers.ollama_embedder.urllib_request.urlopen",
        side_effect=_mock_urlopen(payloads),
    ):
        vector = embedder.embed("hello world")
    assert len(vector) == 3
    assert math.isclose(vector[0], 0.0, rel_tol=1e-6)
    assert math.isclose(vector[1], 0.6, rel_tol=1e-6)
    assert math.isclose(vector[2], 0.8, rel_tol=1e-6)


def test_ollama_embedder_builds_embedding_result_from_request():
    embedder = OllamaEmbedder(model_name="embeddinggemma:latest", normalize=False)
    payloads = [
        {"embedding": [1.0, 0.0, 0.0]},
        {"embedding": [0.0, 1.0, 0.0]},
    ]

    requests = [
        EmbedRequest(
            text="first chunk",
            chunk_id="chunk-1",
            doc_id="doc-1",
            tokens_estimate=5,
            metadata={"content_role": "chunk", "custom": 42},
            is_table=False,
            title="Section 1",
        ),
        EmbedRequest(
            text="second chunk",
            chunk_id="chunk-2",
            doc_id="doc-1",
            tokens_estimate=7,
            metadata={"content_role": "chunk"},
            is_table=True,
            section_path="1.1",
        ),
    ]

    with mock.patch(
        "embedders.providers.ollama_embedder.urllib_request.urlopen",
        side_effect=_mock_urlopen(payloads),
    ):
        results = embedder.embed_batch_req(requests)

    assert len(results) == 2
    first = results[0]
    assert first.chunk_id == "chunk-1"
    assert first.model_name == "embeddinggemma:latest"
    assert first.metadata["title"] == "Section 1"
    assert first.metadata["custom"] == 42

    second = results[1]
    assert second.metadata["is_table"] is True
    assert second.metadata["section_path"] == "1.1"
