"""Dense, sparse, fusion, and cross-encoder retrieval pipeline."""

import math
import re
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from document_ingestion import load_pdf_folder


def create_embedding_model(model_name, local_files_only=True):
    """Create normalized CPU embeddings for cosine-like FAISS comparison."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cpu",
            "local_files_only": local_files_only,
        },
        encode_kwargs={"normalize_embeddings": True},
    )


def create_reranker(model_name, local_files_only=True):
    """Load the cross-encoder used for accurate pairwise relevance scoring."""
    return CrossEncoder(
        model_name,
        device="cpu",
        local_files_only=local_files_only,
    )


def _source_fingerprint(documents_dir, chunk_size, chunk_overlap, model_name):
    """Hash PDF bytes and index settings to detect every meaningful change."""
    folder = Path(documents_dir)
    pdf_files = sorted(folder.rglob("*.pdf")) if folder.exists() else []
    digest = hashlib.sha256()
    digest.update(f"{chunk_size}:{chunk_overlap}:{model_name}".encode())
    for pdf_path in pdf_files:
        digest.update(str(pdf_path.relative_to(folder)).encode("utf-8"))
        with pdf_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest(), pdf_files


def chunk_documents(documents, chunk_size, chunk_overlap):
    """Recursively split extracted modalities while copying all metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for parent_index, document in enumerate(documents):
        pieces = splitter.split_text(document.page_content)
        parent_id = hashlib.sha256(
            f"{document.metadata}|{document.page_content}".encode("utf-8")
        ).hexdigest()[:16]
        for chunk_index, piece in enumerate(pieces):
            metadata = dict(document.metadata)
            metadata.update({
                "parent_id": parent_id,
                "parent_index": parent_index,
                "chunk_index": chunk_index,
                "chunk_count": len(pieces),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            })
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks


def _write_json_atomic(path, value):
    """Atomically replace JSON so interruption cannot corrupt active metadata."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_chunks(path, documents):
    """Persist chunks for BM25 so unchanged runs avoid PDF extraction/chunking."""
    payload = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]
    _write_json_atomic(path, payload)


def _load_chunks(path):
    """Restore the exact chunks associated with a saved FAISS version."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in payload
    ]


def load_or_create_vector_db(
    documents_dir,
    embedding_model,
    embedding_model_name,
    index_root,
    chunk_size,
    chunk_overlap,
):
    """Load an unchanged active index or create a soft-versioned replacement."""
    root = Path(index_root)
    versions_dir = root / "versions"
    manifest_path = root / "manifest.json"
    fingerprint, pdf_files = _source_fingerprint(
        documents_dir,
        chunk_size,
        chunk_overlap,
        embedding_model_name,
    )
    if not pdf_files:
        raise ValueError(f"No PDF files found in the '{documents_dir}' folder.")

    manifest = {"schema_version": 1, "active_version": None, "versions": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Register an index created by the older non-versioned implementation as
    # soft-deleted. Its files remain in place, but it can never become active.
    legacy_exists = (root / "index.faiss").exists() and (root / "index.pkl").exists()
    legacy_registered = any(
        item["id"] == "legacy-unversioned" for item in manifest["versions"]
    )
    if legacy_exists and not legacy_registered:
        manifest["versions"].append({
            "id": "legacy-unversioned",
            "status": "soft_deleted",
            "created_at": None,
            "soft_deleted_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": None,
            "path": ".",
            "chunk_count": None,
            "source_files": [],
        })
        root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(manifest_path, manifest)

    active_id = manifest.get("active_version")
    active = next(
        (item for item in manifest["versions"] if item["id"] == active_id),
        None,
    )
    if active and active["fingerprint"] == fingerprint:
        version_path = root / active["path"]
        vector_db = FAISS.load_local(
            str(version_path),
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        documents = _load_chunks(version_path / "chunks.json")
        return vector_db, documents, active_id, False

    # Parse and recursively chunk only after detecting a changed fingerprint.
    extracted = load_pdf_folder(documents_dir)
    documents = chunk_documents(extracted, chunk_size, chunk_overlap)
    if not documents:
        raise ValueError("PDF files were found, but no searchable content was extracted.")

    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    version_path = versions_dir / version_id
    version_path.mkdir(parents=True, exist_ok=False)
    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local(str(version_path))
    _save_chunks(version_path / "chunks.json", documents)

    # Soft deletion changes status only; all previous files remain recoverable.
    for item in manifest["versions"]:
        if item["status"] == "active":
            item["status"] = "soft_deleted"
            item["soft_deleted_at"] = datetime.now(timezone.utc).isoformat()
    manifest["versions"].append({
        "id": version_id,
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "path": f"versions/{version_id}",
        "chunk_count": len(documents),
        "source_files": [str(path) for path in pdf_files],
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model_name,
    })
    manifest["active_version"] = version_id
    root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(manifest_path, manifest)
    return vector_db, documents, version_id, True


def _tokenize(text):
    """Produce lightweight lowercase terms for local sparse retrieval."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_ranking(documents, query):
    """Rank all documents with BM25 for exact terms, IDs, and names."""
    tokenized_docs = [_tokenize(doc.page_content) for doc in documents]
    query_tokens = _tokenize(query)
    document_count = len(tokenized_docs)
    average_length = sum(map(len, tokenized_docs)) / max(document_count, 1)
    document_frequency = Counter(
        token for tokens in tokenized_docs for token in set(tokens)
    )
    # Standard BM25 defaults balance term frequency and document length.
    k1, b = 1.5, 0.75
    scored = []
    for doc, tokens in zip(documents, tokenized_docs):
        frequencies = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            df = document_frequency[token]
            idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            frequency = frequencies[token]
            denominator = frequency + k1 * (
                1 - b + b * len(tokens) / max(average_length, 1)
            )
            if denominator:
                score += idf * frequency * (k1 + 1) / denominator
        scored.append((doc, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def _reciprocal_rank_fusion(dense_docs, sparse_docs, rank_constant=60):
    """Fuse rankings without comparing incompatible FAISS and BM25 scores."""
    fused = {}
    for ranking in (dense_docs, sparse_docs):
        for rank, doc in enumerate(ranking, start=1):
            key = (doc.page_content, str(doc.metadata))
            if key not in fused:
                fused[key] = [doc, 0.0]
            fused[key][1] += 1.0 / (rank_constant + rank)
    return sorted(fused.values(), key=lambda item: item[1], reverse=True)


def _print_results(title, results, score_name):
    """Print only the requested retrieval diagnostics and complete metadata."""
    print(f"\n================ {title} ================\n")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"{score_name}: {float(score):.4f}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content}")
        print("-" * 60)


def retrieve_and_rerank(
    vector_db,
    all_documents,
    reranker,
    question,
    candidate_k,
    final_top_k,
    dense_max_distance,
    sparse_min_score,
):
    """Run dense + sparse retrieval, RRF fusion, and cross-encoder reranking."""
    # Dense retrieval captures semantic similarity even when wording differs.
    dense_results = [
        (doc, distance)
        for doc, distance in vector_db.similarity_search_with_score(
            question,
            k=candidate_k,
        )
        # FAISS squared L2 distance is inverse: lower means more similar.
        if float(distance) < dense_max_distance
    ]
    # Sparse retrieval protects exact matches such as IDs, acronyms, and names.
    sparse_results = [
        (doc, score)
        for doc, score in _bm25_ranking(all_documents, question)[:candidate_k]
        # BM25 is direct: retain only scores strictly greater than 1.0.
        if float(score) > sparse_min_score
    ]

    _print_results("DENSE RETRIEVAL", dense_results, "FAISS distance")
    _print_results("SPARSE RETRIEVAL", sparse_results, "BM25 score")

    dense_docs = [doc for doc, _ in dense_results]
    sparse_docs = [doc for doc, _ in sparse_results]
    # RRF rewards candidates appearing high in either list without attempting
    # to normalize fundamentally different distance and relevance scales.
    fused = _reciprocal_rank_fusion(dense_docs, sparse_docs)[:candidate_k]
    candidate_docs = [doc for doc, _ in fused]

    # Pairwise reranking is applied only after fusion because running a
    # cross-encoder over every document would be too slow at production scale.
    if not candidate_docs:
        print("\nNo documents passed the dense or sparse retrieval thresholds.")
        return []

    reranker_scores = reranker.predict(
        [(question, doc.page_content) for doc in candidate_docs],
        show_progress_bar=False,
    )
    ranked = sorted(
        zip(candidate_docs, reranker_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:final_top_k]

    # Store the final score in metadata for observability and later debugging.
    for doc, score in ranked:
        doc.metadata["reranker_score"] = float(score)

    # These are the evidence passages supplied to Gemini. They are displayed
    # for retrieval debugging and must not be mistaken for generated answers.
    _print_results(
        f"TOP {len(ranked)} RERANKED CONTEXT CHUNKS",
        ranked,
        "Reranker score",
    )
    return [doc for doc, _ in ranked]
