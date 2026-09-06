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
    def __init__(self, api_key, model, embedding_model):
        chat_model = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0,
        )
        self.llm = LangchainLLMWrapper(chat_model)
        self.embeddings = LangchainEmbeddingsWrapper(embedding_model)

        # Groq permits one completion per request.
        answer_relevancy.strictness = 1

    @staticmethod
    def _print_result(title, result):
        print(f"\n{'=' * 12} {title} {'=' * 12}\n")
        print(result)
        print(result.to_pandas().to_string(index=False))

    def evaluate_retrieval(self, question, retrieved_contexts, reference_answer):
        dataset = Dataset.from_dict({
            "question": [question],
            "contexts": [retrieved_contexts],
            "ground_truth": [reference_answer],
        })
        result = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall],
            llm=self.llm,
        )
        self._print_result("RAGAS: AFTER VECTOR DB", result)
        return result

    def build_generation_dataset(
        self, question, answer, retrieved_contexts, reference_answer
    ):
        return Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [retrieved_contexts],
            "ground_truth": [reference_answer],
        })

    def evaluate_generation(self, dataset):
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=self.llm,
            embeddings=self.embeddings,
        )
        self._print_result("RAGAS: AFTER LLM", result)
        return result

    def evaluate_full(self, dataset):
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
        )
        self._print_result("FULL RAGAS EVALUATION", result)
        return result
