from chunkers.model import Chunk, ChunkSet, ChunkStrategy, ChunkType, ProvenanceAgg
from chunkers.model.block_span import BlockSpan
from embedders import ChunkSetEmbedder
from embedders.i_embedder import IEmbedder
from embedders.model.embedding_result import EmbeddingResult


class StubEmbedder(IEmbedder):
    """Simple embedder for testing that returns deterministic vectors."""

    def __init__(self):
        self.model_name = "stub/embedding"
        self.device = "cpu"
        self.max_tokens = 4096

    def get_dimensions(self) -> int:
        return 3

    def embed(self, text: str):
        return [float(len(text)), 1.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]

    def embed_request(self, req):
        return self._build_result(req)

    def embed_batch_req(self, reqs):
        return [self._build_result(req) for req in reqs]

    def _build_result(self, req):
        return EmbeddingResult(
            chunk_id=req.chunk_id,
            embedding=self.embed(req.text),
            text_embedded=req.text,
            token_count=req.tokens_estimate or len(req.text) // 4,
            model_name=self.model_name,
            metadata=req.metadata,
            doc_id=req.doc_id,
            lang=req.lang,
        )


def _make_chunk_set() -> ChunkSet:
    provenance = ProvenanceAgg(
        source_blocks=["block-1"],
        spans=[
            BlockSpan(block_id="block-1", start_char=0, end_char=10, page_number=1)
        ],
        page_numbers={1},
        doc_id="doc-001",
        file_path="/tmp/sample.pdf",
        metadata={"confidence": 0.95},
    )
    chunk = Chunk(
        chunk_id="chunk-1",
        text="Sample chunk text for embedding.",
        token_count=8,
        char_count=32,
        chunk_type=ChunkType.HYBRID,
        strategy=ChunkStrategy.SEMANTIC_COHERENCE,
        provenance=provenance,
        metadata={
            "group_type": "table",
            "table_title": "Sales Overview",
            "table_caption": "Quarterly revenue in USD.",
            "lang": "vi",
            "hash": "abc123",
            "section_path": "1.2",
        },
        section_title="Financial Summary",
    )
    chunk_set = ChunkSet(
        doc_id="doc-001",
        chunks=[chunk],
        file_path="/tmp/sample.pdf",
        total_tokens=8,
        total_chars=32,
        chunk_strategy="hybrid",
        metadata={"source": "unit-test"},
    )
    return chunk_set


def test_chunkset_embedder_builds_requests_with_provenance():
    chunk_set = _make_chunk_set()
    embedder = ChunkSetEmbedder(
        embedder=StubEmbedder(),
        include_titles=True,
        include_table_metadata=True,
        include_captions=True,
    )

    results = embedder.embed_chunk_set(chunk_set)

    # Expect chunk + title + table title + caption
    assert len(results) == 4
    chunk_result = next(res for res in results if res.chunk_id == "chunk-1")

    assert chunk_result.metadata["content_role"] == "chunk"
    assert chunk_result.metadata["provenance"]["page_numbers"] == [1]
    assert chunk_result.metadata["chunk_hash"] == "abc123"

    table_title_result = next(res for res in results if res.chunk_id.endswith("table_title"))
    assert table_title_result.metadata["content_role"] == "table_title"
    assert table_title_result.metadata["provenance"]["source_blocks"] == ["block-1"]
    assert table_title_result.lang == "vi"
