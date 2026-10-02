GENERATOR_PROMPT = """
You are a helpful assistant answering questions based only on the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Answer the question using only the information provided in the context.
- Do not use outside knowledge.
- If the answer cannot be found in the context, say:
  "I don't know based on the provided context."
- Be concise and directly answer the question.
- Do not mention the retrieval process or the context.

Answer:
"""
