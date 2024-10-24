RAG_PROMPT = """Your task is to answer a question based on the realted documents and your own knowledge.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Only output the answer related to the question.
Use three sentences maximum and keep the answer as concise as possible. 

Related Documents:
{context}

Question: {question}

Helpful Answer:"""

RAG_PROMPT_FEW_SHOT = """Your task is to answer a question based on the realted documents and your own knowledge.
If you believe there are no related dodcuments to the question, answer the question based on your own knowledge base.
Only output the answer directly related to the question, like a phrase or a sentence.
Keep the answer as concise as possible. 

Related Documents:
# event_name
Yalda Night
# event_time
Dec 20,2024|All Day

Question: When is Yalda Night held?

Helpful Answer: Dec 20, 2024.

Related Documents:
Governor Dinwiddie sent Captain William Trent to build a fort at the Forks of the Ohio On February 17 1754 Trent began construction of the fort the first European habitation17 at the site of presentday Pittsburgh The fort named Fort Prince George was only halfbuilt by April 1754 when over 500 French forces arrived and ordered the 40some colonials back to Virginia The French tore down the British fortification and constructed Fort Duquesne1415

Question: What was the first European-built fort at the site of present-day Pittsburgh?

Helpful Answer: Fort Prince George

Related Documents:
{context}

Question: {question}\n\nHelpful Answer: """