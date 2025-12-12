INVALID_UNICODE_CLEANUP_REGEX = r'[\p{Cf}\p{Cn}\p{Co}\p{Cs}\p{So}]'
VECTOR_EMBEDDINGS_SIMILARITY_THRESHOLD = 1.15
VECTOR_EMBEDDINGS_QUERY_SYSTEM_PROMPT = "You are an assistant for a naturopathic medicine clinic. For general questions, provide a brief, high-level summary as a reply but avoid long answers. Provide more detail if the user asks specific follow-up questions. Answer the question based only on the following context: {context}\n\nDo not tell the user to contact the clinic in your answer, simply provide the information requested.\n\nQuestion: {question}"
