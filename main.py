"""Application entry point that orchestrates PDF RAG and evaluation."""

from config import (
    CANDIDATE_K,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FINAL_TOP_K,
    GENERATION_MODEL,
    GROQ_API_KEY,
    RERANKER_MODEL,
)
from document_ingestion import load_pdf_folder
from llm_service import generate_answer
from ragas_evaluator import RagasEvaluator
from vector_pipeline import (
    create_embedding_model,
    create_reranker,
    create_vector_db,
    retrieve_and_rerank,
)


def ask_non_empty(prompt):
    """Request a required interactive value and reject blank input."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("This value cannot be empty.")


def main():
    """Run ingestion, hybrid retrieval, generation, and RAGAS evaluation."""
    # A reference answer is requested because context precision and recall
    # cannot be measured correctly without expected evidence.
    question = ask_non_empty("Enter your question: ")
    reference_answer = ask_non_empty("Enter the RAGAS reference answer: ")

    # PDF ingestion produces independent text, table, flow, image, and chart
    # documents while retaining source and page metadata.
    documents = load_pdf_folder(DOCUMENTS_DIR)
    if not documents:
        raise ValueError(f"No PDF content found in the '{DOCUMENTS_DIR}' folder.")

    # The embedding model powers dense search. The cross-encoder is reserved
    # for the much smaller fused candidate set to keep reranking affordable.
    embedding_model = create_embedding_model(EMBEDDING_MODEL)
    reranker = create_reranker(RERANKER_MODEL)
    # Rebuild the index from the current Documents folder so changed PDFs do
    # not leave stale chunks in the active search index.
    vector_db = create_vector_db(
        documents,
        embedding_model,
        FAISS_INDEX_PATH,
    )
    # Hybrid retrieval combines semantic and exact-term matches before the
    # cross-encoder selects the final context.
    retrieved_docs = retrieve_and_rerank(
        vector_db,
        documents,
        reranker,
        question,
        CANDIDATE_K,
        FINAL_TOP_K,
    )
    retrieved_contexts = [doc.page_content for doc in retrieved_docs]

    # RAGAS retrieval metrics are calculated from the exact context that will
    # subsequently be supplied to the generation model.
    evaluator = RagasEvaluator(
        GROQ_API_KEY,
        GENERATION_MODEL,
        embedding_model,
    )
    retrieval_result = evaluator.evaluate_retrieval(
        question,
        retrieved_contexts,
        reference_answer,
    )
    # Generate only after retrieval evaluation so the two pipeline stages
    # remain independently measurable.
    answer = generate_answer(
        GROQ_API_KEY,
        GENERATION_MODEL,
        question,
        retrieved_docs,
    )
    # Generation metrics require the answer, while the complete evaluation
    # reports all retrieval and generation metrics together.
    generation_dataset = evaluator.build_generation_dataset(
        question,
        answer,
        retrieved_contexts,
        reference_answer,
    )
    generation_result = evaluator.evaluate_generation(generation_dataset)
    ragas_result = evaluator.evaluate_full(generation_dataset)

    print("\n================ FINAL ANSWER ================\n")
    print(answer)
    print("\n================ RAGAS RESULTS ================\n")
    print(f"After Vector DB: {retrieval_result}")
    print(f"After LLM: {generation_result}")
    print(f"Full evaluation: {ragas_result}")


if __name__ == "__main__":
    # This guard prevents the expensive pipeline from running during imports.
    main()
