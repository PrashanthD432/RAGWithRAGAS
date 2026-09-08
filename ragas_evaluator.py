"""Stage-specific and complete RAGAS evaluation helpers."""

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import aevaluate
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
        chat_model = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            timeout=30,
            max_retries=1,
        )
        self.llm = LangchainLLMWrapper(chat_model)
        self.embeddings = LangchainEmbeddingsWrapper(embedding_model)

        # One generated reverse-question limits judge calls and quota usage.
        answer_relevancy.strictness = 1

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

    async def evaluate_full(self, dataset):
        """Asynchronously run all four metrics after answer generation."""
        result = await aevaluate(
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

    @staticmethod
    def split_full_result(result):
        """Create concise retrieval/generation views without rerunning metrics."""
        row = result.to_pandas().iloc[0]
        retrieval = {
            "context_precision": row.get("context_precision"),
            "context_recall": row.get("context_recall"),
        }
        generation = {
            "faithfulness": row.get("faithfulness"),
            "answer_relevancy": row.get("answer_relevancy"),
        }
        return retrieval, generation
