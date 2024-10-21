# docs with different types (json, pdf, csv) 
# input: raw file path;
# output: a retrieval

# TODO ####################################################################################
# step1: generate summaries for table; reference: https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb
# step2: retriever = MultiVectorRetriever
# step3: splitter
# step4: save_faiss_multi_vector_index
###########################################################################################

import os
import uuid
import json
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def json_to_markdown_documents(json_file_path):
    documents = []
    
    def json_to_markdown_string(json_data, level=1):
        markdown_string = ""
        for key, value in json_data.items():
            markdown_string += f"{'#' * level} {key}\n\n"
            if isinstance(value, dict):
                markdown_string += json_to_markdown_string(value, level + 1)
            elif isinstance(value, list):
                for item in value:
                    markdown_string += f"- {item}\n"
            else:
                markdown_string += f"{value}\n\n"
        return markdown_string
    
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
        # Loop through each key and value to create separate Documents
        if isinstance(json_data, list):
            for json_item in json_data:
                markdown_content = json_to_markdown_string(json_item)
                assert isinstance(markdown_content, str)
                documents.append(Document(page_content =  markdown_content))
        elif isinstance(json_data, dict):
            for key, value in json_data.items():
                markdown_content = json_to_markdown_string({key: value})
                assert isinstance(markdown_content, str)
                documents.append(Document(page_content =  markdown_content))
        
    return documents

def load_documents(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path)
    elif file_type == "json":
        # 转markdown -> 返回List[Document] 
        return json_to_markdown_documents(file_path) # return: List[Document]
    else:
        raise ValueError("Unsupported file type")
    
    return loader.load() # return: List[Document]

def create_embedding(args):
    embedding_model_kwargs = {'device': args.device}
    embedding_encode_kwargs = {
        'batch_size': args.batch_size,
        'normalize_embeddings': args.normalize_embeddings
    }

    # Initialize embeddings
    local_embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs,
        show_progress=args.show_progress_bar  
    )
    return local_embeddings

def save_faiss_multi_vector_index(args):
    # Initialize embeddings
    local_embeddings = create_embedding(args)

    all_docs = []

    # BUILD DOCUMENTS ----
    for root, _, files in os.walk(args.directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_type = file.split('.')[-1].lower()
            if file_type not in ["pdf", "csv", "json"]:
                continue
            try:
                docs = load_documents(file_path, file_type) #  List[Document]
                # Chunking ----
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
                splits = text_splitter.split_documents(docs)  
                all_docs.extend(splits)
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

    # Create FAISS retriever and save the index ----
    # all_docs: List[Document] 
    # local_embeddings: HuggingFaceEmbeddings
    faiss_retriever = FAISS.from_documents(all_docs, local_embeddings) 
    faiss_retriever.save_local(args.faiss_output_dir)

    return faiss_retriever
