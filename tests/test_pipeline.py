from pathlib import Path

from fastapi.testclient import TestClient

from datheon.api import app
from datheon import ingestion
from datheon.ingestion import chunk_documents, load_documents, load_manifest
from datheon.rag import User, citations_for, retrieve
from datheon.vector_store import ChromaVectorStore, DeterministicTestEmbedder


ROOT = Path(__file__).resolve().parents[1]


def sample_store() -> ChromaVectorStore:
    documents = load_documents(ROOT / "sample-data", load_manifest(ROOT / "sample-data/manifest.jsonl"))
    chunks = chunk_documents(documents)
    store = ChromaVectorStore(":memory:", "test_documents", DeterministicTestEmbedder())
    store.upsert(chunks)
    return store


def test_loads_supported_formats_and_preserves_metadata():
    manifest = load_manifest(ROOT / "sample-data/manifest.jsonl")
    documents = load_documents(ROOT / "sample-data", manifest)
    chunks = chunk_documents(documents)
    assert {Path(item.metadata["source"]).suffix for item in documents} == {".md", ".csv"}
    assert all({"department", "access_tier", "date", "source"} <= item["metadata"].keys() for item in chunks)
    assert len({item["id"] for item in chunks}) == len(chunks)


def test_pdf_pages_keep_page_metadata(tmp_path, monkeypatch):
    class FakePage:
        def extract_text(self):
            return "A PDF page with searchable content."

    class FakeReader:
        pages = [FakePage()]

    (tmp_path / "policy.pdf").write_bytes(b"mock pdf")
    monkeypatch.setattr(ingestion, "PdfReader", lambda _: FakeReader())
    documents = load_documents(
        tmp_path,
        {"policy.pdf": {"department": "Legal", "access_tier": "restricted", "date": "2025-01-01"}},
    )
    assert documents[0].text == "A PDF page with searchable content."
    assert documents[0].metadata["page"] == 1
    assert documents[0].metadata["department"] == "Legal"


def test_filter_enforces_department_and_tier_before_returning_sources():
    store = sample_store()
    assert store.collection.metadata["hnsw:space"] == "cosine"
    junior = User("Finance", ("public",))
    chunks = retrieve(store, "expense approval limit forecast", junior)
    assert chunks
    assert all(chunk["metadata"]["department"] == "Finance" for chunk in chunks)
    assert all(chunk["metadata"]["access_tier"] == "public" for chunk in chunks)
    assert all("forecast" not in chunk["text"].lower() for chunk in chunks)


def test_cross_department_and_confidential_isolation():
    store = sample_store()
    engineering = retrieve(store, "expense approval limit", User("Engineering", ("public", "restricted")))
    assert engineering
    assert all(chunk["metadata"]["department"] == "Engineering" for chunk in engineering)
    assert not any("approval_limit" in chunk["text"] for chunk in engineering)


def test_citations_are_derived_from_retrieved_chunks():
    chunks = retrieve(sample_store(), "on-call production alert", User("Engineering", ("public",)))
    citations = citations_for(chunks)
    assert citations
    assert all(item["department"] == "Engineering" and item["chunk_id"] for item in citations)


def test_api_rejects_missing_or_invalid_tokens(monkeypatch):
    from datheon import api

    monkeypatch.setattr(api, "store", sample_store())
    monkeypatch.setattr(api, "api_keys", {"known": User("Engineering", ("public",))})
    client = TestClient(app)
    assert client.post("/query", json={"question": "alert"}).status_code == 401
    assert client.post("/query", headers={"Authorization": "Bearer unknown"}, json={"question": "alert"}).status_code == 401
    response = client.post("/query", headers={"Authorization": "Bearer known"}, json={"question": "alert"})
    assert response.status_code == 200
    assert response.json()["sources"]