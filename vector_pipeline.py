"""Dense, sparse, fusion, and cross-encoder retrieval pipeline."""

import math
import re
from collections import Counter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


def create_embedding_model(model_name):
    """Create normalized CPU embeddings for cosine-like FAISS comparison."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_reranker(model_name):
    """Load the cross-encoder used for accurate pairwise relevance scoring."""
    return CrossEncoder(model_name, device="cpu")


def create_vector_db(documents, embedding_model, index_path):
    """Build and persist a fresh FAISS index from the current PDF chunks."""
    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local(index_path)
    return vector_db


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
):
    """Run dense + sparse retrieval, RRF fusion, and cross-encoder reranking."""
    # Dense retrieval captures semantic similarity even when wording differs.
    dense_results = vector_db.similarity_search_with_score(
        question,
        k=candidate_k,
    )
    # Sparse retrieval protects exact matches such as IDs, acronyms, and names.
    sparse_results = _bm25_ranking(all_documents, question)[:candidate_k]

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

    _print_results("FINAL RERANKED RESULT", ranked, "Reranker score")
    return [doc for doc, _ in ranked]
