# FastAPI app for RAG server

import os
from langsmith import Client
from langserve_server import add_routes
from langserve_server import app
from langchain_core.prompts import PromptTemplate
from prompt import *
import uvicorn
os.environ['LANGCHAIN_API_KEY'] = 'ls__LANGCHAIN_API_KEY__'


client = Client() # langsmith for monitor

################################# RAG CHAIN  ################################## 
############################# TODO: Add retriever ############################# 
custom_rag_prompt = PromptTemplate.from_template(RAG_PROMPT)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
############################# TODO: Add retriever ############################# 

add_routes(
    app,
    rag_chain,
    path="/rag"
)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
 
