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
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path)
    elif file_type == "json":
        loader = JSONLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    return loader.load()

def save_faiss_multi_vector_index(args):
    # Set default values
    embedding_model_kwargs = {'device': args.device}
    embedding_encode_kwargs = {
        'batch_size': args.batch_size,
        'normalize_embeddings': args.normalize_embeddings,
        'show_progress_bar': args.show_progress_bar
    }


    # Initialize embeddings
    local_embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )

    all_docs = []

    # BUILD DOCUMENTS!
    for root, _, files in os.walk(args.directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_type = file.split('.')[-1].lower()
            if file_type not in ["pdf", "csv", "json"]:
                continue
            try:
                docs = load_documents(file_path, file_type)

                # Chunking ----
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
                splits = text_splitter.split_documents(docs)  
                all_docs.extend(splits)
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

    # Create FAISS retriever and save the index
    faiss_retriever = FAISS.from_documents(all_docs, local_embeddings)
    faiss_retriever.save_local(args.output_dir)

    return faiss_retriever
