"""Stage-specific and complete RAGAS evaluation helpers."""

from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


class RagasEvaluator:
    """Evaluate retrieval and generation with shared judge dependencies."""

    def __init__(self, api_key, model, embedding_model):
        """Create one reusable judge LLM and embedding adapter."""
        # Reusing these wrappers prevents repeated model initialization for
        # the retrieval, generation, and complete evaluation stages.
        chat_model = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0,
        )
        self.llm = LangchainLLMWrapper(chat_model)
        self.embeddings = LangchainEmbeddingsWrapper(embedding_model)

        # Groq permits one completion per request.
        answer_relevancy.strictness = 1

    def evaluate_retrieval(self, question, retrieved_contexts, reference_answer):
        """Measure context precision and recall immediately after retrieval."""
        # This dataset intentionally has no generated answer because these
        # metrics evaluate the vector/hybrid retrieval stage by itself.
        dataset = Dataset.from_dict({
            "question": [question],
            "contexts": [retrieved_contexts],
            "ground_truth": [reference_answer],
        })
        result = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall],
            llm=self.llm,
            show_progress=False,
        )
        return result

    def build_generation_dataset(
        self, question, answer, retrieved_contexts, reference_answer
    ):
        """Build the common dataset required by answer-level RAGAS metrics."""
        return Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [retrieved_contexts],
            "ground_truth": [reference_answer],
        })

    def evaluate_generation(self, dataset):
        """Measure whether the LLM answer is grounded and question-relevant."""
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=self.llm,
            embeddings=self.embeddings,
            show_progress=False,
        )
        return result

    def evaluate_full(self, dataset):
        """Run all four metrics together after answer generation."""
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            llm=self.llm,
            embeddings=self.embeddings,
            show_progress=False,
        )
        return result
