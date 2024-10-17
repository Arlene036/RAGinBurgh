# FastAPI app for RAG server

import os
from langsmith import Client
from langserve_server import add_routes
from langserve_server import app
from langchain_core.prompts import PromptTemplate
from prompt import *
import uvicorn
from fastapi import FastAPI
from rag import PittsRAG

os.environ['LANGCHAIN_API_KEY'] = 'ls__LANGCHAIN_API_KEY__'


client = Client() # langsmith for monitor
app = FastAPI()

################################# RAG CHAIN TODO  ################################## 
rag = PittsRAG(generator = MODEL, retrieval = RETRIEVAL)

def run_rag(input: Input):
    try:
        result = rag.inference(input) 
        return result
    except Exception as e:
        print(e)

######################################################################## 

add_routes(
    app,
    RunnableLambda(run_rag),
    path="/rag"
)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
 
