"""Grounded answer generation through Gemini's Interactions API."""

from google import genai
from google.genai.errors import ClientError


class LLMRateLimitError(RuntimeError):
    """Raised when Gemini has no remaining request or token capacity."""


class LLMModelNotFoundError(RuntimeError):
    """Raised when the configured Gemini model is unavailable."""


class LLMConnectionError(RuntimeError):
    """Raised when the Gemini service cannot be reached."""


def build_context(retrieved_docs):
    """Serialize reranked documents while preserving source provenance."""
    # Preserve provenance for every retrieved chunk so the model can cite the
    # correct source page and distinguish prose, tables, charts, and flows.
    return "\n\n".join(
        (
            f"[Source {index}]\n"
            f"File: {doc.metadata.get('source', 'unknown')}\n"
            f"Page: {doc.metadata.get('page', 'unknown')}\n"
            f"Content type: {doc.metadata.get('content_type', 'text')}\n"
            f"Header: {doc.metadata.get('header')}\n"
            f"Section: {doc.metadata.get('section')}\n\n"
            f"Content:\n{doc.page_content}"
        )
        for index, doc in enumerate(retrieved_docs, start=1)
    )


async def generate_answer(api_key, model, question, retrieved_docs):
    """Asynchronously generate an answer constrained to retrieved evidence."""
    # Avoid an unnecessary model call when retrieval rejects every document.
    if not retrieved_docs:
        return "I don't have enough information in the provided documents."

    # The generation model receives only the final reranked evidence, never the
    # complete document collection. This reduces noise and limits hallucination.
    context = build_context(retrieved_docs)
    prompt = f"""
You are a document question-answering assistant.

Answer the question ONLY using the context below.
Synthesize all relevant source chunks into one cohesive final answer. Do not
write a separate answer for each source chunk and do not repeat information.
Do not infer, invent, or add process steps that are not explicitly present.
When a retrieved chunk contains "Exact flow from the diagram", reproduce its
step names and order exactly. Do not replace that flow with related prose.
For tables, preserve the relationship between column names and cell values.
End factual statements with source references in the form [Source N].

If the answer is not available in the context, say:
"I don't have enough information in the provided documents."

Context:
-------------------------
{context}
-------------------------

Question:
{question}

Answer:
"""
    # Google recommends the Interactions API for Gemini 3.x. Using it directly
    # also avoids the generate_content automatic-function-calling warning.
    client = genai.Client(api_key=api_key)
    try:
        response = await client.aio.interactions.create(
            model=model,
            input=prompt,
            timeout=30,
        )
    except ClientError as error:
        status_code = getattr(error, "code", None)
        detail = getattr(error, "message", str(error))
        if status_code == 429:
            raise LLMRateLimitError(
                f"Gemini rate limit reached. {detail}"
            ) from error
        if status_code == 404:
            raise LLMModelNotFoundError(
                f"Gemini model '{model}' is unavailable. {detail}"
            ) from error
        raise
    except Exception as error:
        # The Interactions client currently exposes its connection exception
        # from an internal module, so detect that stable exception name without
        # coupling application code to a private import path.
        if type(error).__name__ == "APIConnectionError":
            raise LLMConnectionError(
                "Could not connect to Gemini. Check internet, proxy, and firewall."
            ) from error
        raise
    finally:
        await client.aio.aclose()
    return response.output_text or ""
