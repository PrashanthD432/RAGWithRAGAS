from groq import Groq


def build_context(retrieved_docs):
    return "\n\n".join(
        (
            f"Header: {doc.metadata.get('header')}\n"
            f"Section: {doc.metadata.get('section')}\n\n"
            f"Content:\n{doc.page_content}"
        )
        for doc in retrieved_docs
    )


def generate_answer(api_key, model, question, retrieved_docs):
    context = build_context(retrieved_docs)
    prompt = f"""
You are a tax assistant.

Answer the question ONLY using the context below.

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
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only retrieved document context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content
