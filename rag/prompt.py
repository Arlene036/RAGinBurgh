RAG_PROMPT = """Your task is to answer a question based on the realted documents and your own knowledge.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible. Only output the answer related to the question.

Related Documents:
{context}

Question: {question}

Helpful Answer:"""

RAG_PROMPT_FEW_SHOT = """Your task is to answer a question based on the realted documents and your own knowledge.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible. Only output the answer related to the question.

Related Documents:
# event_name
Yalda Night
# event_time
Dec 20,2024|All Day

Question: When is Yalda Night held?

Helpful Answer: Yalda Night is held on Dec 20, 2024.

Related Documents:
# event_name
Make-Up Final Exams
# event_time
Dec 16,2024|All Day

Question: When does Make-Up Final Exams in 2024 happen?

Helpful Answer: Make-Up Final Exams will happen on Dec 16,2024.

Related Documents:
{context}

Question: {question}

Helpful Answer:

"""