from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    GENERATION_MODEL,
    GROQ_API_KEY,
    TOP_K,
)
from corpus import DOCUMENTS
from llm_service import generate_answer
from ragas_evaluator import RagasEvaluator
from vector_pipeline import (
    create_embedding_model,
    create_vector_db,
    demonstrate_embeddings,
    prepare_section_documents,
    print_chunks,
    retrieve_documents,
    split_documents,
)


def ask_non_empty(prompt):
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("This value cannot be empty. Please try again.")


def get_evaluation_inputs():
    print("\n================ USER INPUT =================\n")
    question = ask_non_empty("Enter your question: ")
    reference_answer = ask_non_empty(
        "Enter the expected/reference answer for RAGAS evaluation: "
    )
    return question, reference_answer


def main():
    question, reference_answer = get_evaluation_inputs()

    chunks = split_documents(DOCUMENTS)
    print_chunks(chunks)

    embedding_model = create_embedding_model(EMBEDDING_MODEL)
    demonstrate_embeddings(chunks, embedding_model)

    section_documents = prepare_section_documents(chunks)
    vector_db = create_vector_db(
        section_documents,
        embedding_model,
        FAISS_INDEX_PATH,
    )
    retrieved_docs = retrieve_documents(vector_db, question, TOP_K)
    retrieved_contexts = [doc.page_content for doc in retrieved_docs]

    evaluator = RagasEvaluator(
        GROQ_API_KEY,
        GENERATION_MODEL,
        embedding_model,
    )

    # After Vector DB: Context Precision and Context Recall
    retrieval_result = evaluator.evaluate_retrieval(
        question,
        retrieved_contexts,
        reference_answer,
    )

    answer = generate_answer(
        GROQ_API_KEY,
        GENERATION_MODEL,
        question,
        retrieved_docs,
    )
    print("\n================ FINAL ANSWER =================\n")
    print(answer)

    generation_dataset = evaluator.build_generation_dataset(
        question,
        answer,
        retrieved_contexts,
        reference_answer,
    )

    # After LLM: Faithfulness and Answer Relevancy
    generation_result = evaluator.evaluate_generation(generation_dataset)

    # Full RAGAS evaluation: all four metrics, run after the LLM
    ragas_result = evaluator.evaluate_full(generation_dataset)

    return retrieval_result, generation_result, ragas_result


if __name__ == "__main__":
    main()
