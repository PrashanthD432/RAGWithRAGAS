"""Grounded prompt construction and Groq answer generation."""

from groq import Groq


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


def generate_answer(api_key, model, question, retrieved_docs):
    """Generate a deterministic answer constrained to retrieved evidence."""
    # The generation model receives only the final reranked evidence, never the
    # complete document collection. This reduces noise and limits hallucination.
    context = build_context(retrieved_docs)
    prompt = f"""
You are a document question-answering assistant.

Answer the question ONLY using the context below.
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
    # Bounded retries and timeout prevent requests from hanging indefinitely.
    client = Groq(api_key=api_key, timeout=30.0, max_retries=2)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only retrieved document context.",
            },
            {"role": "user", "content": prompt},
        ],
        # Deterministic generation is preferable for factual RAG evaluation.
        temperature=0,
    )
    return response.choices[0].message.content
