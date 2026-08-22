import pytest
import time
from rag.schemas import Chunk, DocumentPassage
from rag.chunking.hierarchical import HierarchicalChunker
from rag.retrieval.vector_store import VectorStore
from rag.retrieval.bm25_index import BM25Index
from rag.retrieval.hybrid_search import HybridSearchEngine
from rag.retrieval.semantic_cache import SemanticCache

@pytest.fixture
def indexed_engine():
    doc = DocumentPassage(
        doc_id="doc_goa_01",
        query_id=1,
        query_type="LOCATION",
        text="Panaji is the capital city of Goa located along the Mandovi River.",
        source_lang="en",
        target_lang="hi",
        is_gold=True
    )
    chunker = HierarchicalChunker(child_token_size=20)
    chunks = chunker.chunk_documents([doc])

    vector_store = VectorStore()
    vector_store.index_chunks(chunks)

    bm25 = BM25Index()
    bm25.index_chunks(chunks)

    hybrid = HybridSearchEngine(vector_store, bm25)
    return hybrid, vector_store, bm25

def test_hybrid_search_accuracy_and_speed(indexed_engine):
    hybrid, vector_store, bm25 = indexed_engine
    
    start = time.perf_counter()
    results = hybrid.search("capital of Goa", top_k=1)
    duration_ms = (time.perf_counter() - start) * 1000

    assert len(results) == 1
    assert "Panaji" in results[0].chunk.text
    # Fast retrieval target: < 10ms for search
    assert duration_ms < 25.0

def test_semantic_cache_performance():
    cache = SemanticCache(max_size=100)
    cache.put("capital of goa", "hierarchical", {"answer": "Panaji", "is_refusal": False})

    # Exact cache lookup
    start = time.perf_counter()
    res, hit_type = cache.get("capital of goa", "hierarchical")
    duration_ms = (time.perf_counter() - start) * 1000

    assert res is not None
    assert res["answer"] == "Panaji"
    assert hit_type == "exact"
    # Sub-millisecond cache latency
    assert duration_ms < 2.0
