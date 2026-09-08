"""Application entry point that orchestrates PDF RAG and evaluation."""

import asyncio

from config import (
    CANDIDATE_K,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    DENSE_MAX_DISTANCE,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FINAL_TOP_K,
    GENERATION_MODEL,
    GEMINI_API_KEY,
    HF_LOCAL_FILES_ONLY,
    RERANKER_MODEL,
    RUN_RAGAS,
    SPARSE_MIN_SCORE,
)
from llm_service import (
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    generate_answer,
)
from ragas_evaluator import RagasEvaluator
from vector_pipeline import (
    create_embedding_model,
    create_reranker,
    load_or_create_vector_db,
    retrieve_and_rerank,
)


async def ask_non_empty(prompt):
    """Request input without blocking the asyncio event loop."""
    while True:
        # Terminal input is blocking, so move it to a worker thread.
        value = (await asyncio.to_thread(input, prompt)).strip()
        if value:
            return value
        print("This value cannot be empty.")


async def main():
    """Asynchronously orchestrate ingestion, retrieval, generation, and evaluation."""
    # The user question is always required for retrieval and generation.
    question = await ask_non_empty("Enter your question: ")
    # A reference answer is needed only when the optional RAGAS judge is active.
    reference_answer = None
    if RUN_RAGAS:
        reference_answer = await ask_non_empty("Enter the RAGAS reference answer: ")

    # Local model loading is blocking, so initialize both models concurrently
    # without blocking the event loop.
    embedding_task = asyncio.to_thread(
        create_embedding_model,
        EMBEDDING_MODEL,
        HF_LOCAL_FILES_ONLY,
    )
    reranker_task = asyncio.to_thread(
        create_reranker,
        RERANKER_MODEL,
        HF_LOCAL_FILES_ONLY,
    )
    embedding_model, reranker = await asyncio.gather(
        embedding_task,
        reranker_task,
    )

    # Fingerprinting avoids PDF parsing and chunking when neither the documents
    # nor the chunk/model configuration has changed.
    vector_db, documents, index_version, rebuilt = await asyncio.to_thread(
        load_or_create_vector_db,
        DOCUMENTS_DIR,
        embedding_model,
        EMBEDDING_MODEL,
        FAISS_INDEX_PATH,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    index_action = "created" if rebuilt else "loaded"
    print(f"Index version {index_version} {index_action}; chunks: {len(documents)}")
    # Hybrid retrieval combines semantic and exact-term matches before the
    # cross-encoder selects the final context.
    retrieved_docs = await asyncio.to_thread(
        retrieve_and_rerank,
        vector_db,
        documents,
        reranker,
        question,
        CANDIDATE_K,
        FINAL_TOP_K,
        DENSE_MAX_DISTANCE,
        SPARSE_MIN_SCORE,
    )
    retrieved_contexts = [doc.page_content for doc in retrieved_docs]

    # Answer generation is the only Gemini call during normal application use.
    try:
        answer = await generate_answer(
            GEMINI_API_KEY,
            GENERATION_MODEL,
            question,
            retrieved_docs,
        )
    except (
        LLMConnectionError,
        LLMRateLimitError,
        LLMModelNotFoundError,
    ) as error:
        print(f"\n{error}")
        print(
            "The retrieved results above are still valid. "
            "Resolve the Gemini service issue and retry."
        )
        return

    print("\n================ FINAL ANSWER ================\n")
    print(answer)

    if RUN_RAGAS:
        # Run each metric exactly once. Stage-specific summaries are derived
        # from the full result instead of repeating costly judge-model calls.
        evaluator = RagasEvaluator(
            GEMINI_API_KEY,
            GENERATION_MODEL,
            embedding_model,
        )
        generation_dataset = evaluator.build_generation_dataset(
            question,
            answer,
            retrieved_contexts,
            reference_answer,
        )
        ragas_result = await evaluator.evaluate_full(generation_dataset)
        retrieval_result, generation_result = evaluator.split_full_result(
            ragas_result
        )

        print("\n================ RAGAS RESULTS ================\n")
        print(f"After Vector DB: {retrieval_result}")
        print(f"After LLM: {generation_result}")
        print(f"Full evaluation: {ragas_result}")


if __name__ == "__main__":
    # This guard prevents the expensive pipeline from running during imports.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
