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
from langchain_community.document_loaders.pdf_loader import PDFLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(file_path, file_type):
    if file_type == "pdf":
        loader = PDFLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path)
    elif file_type == "json":
        loader = JSONLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    return loader.load()

def save_faiss_multi_vector_index(embedding_model_name, directory_path, output_dir):
    embedding_model_kwargs = {'device': 'cuda:0'} # if gpu
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False} # TODO batch_size?
    local_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )

    all_docs = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_type = file.split('.')[-1].lower() 
            if file_type not in ["pdf", "csv", "json"]: # CHECK
                continue
            try:
                docs = load_documents(file_path, file_type)

                # Chunking ---
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)  
                all_docs.extend(splits) 
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

    faiss_retriever = FAISS.from_documents(all_docs, local_embeddings)
    faiss_retriever.save_local(output_dir)

    return faiss_retriever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process documents and save FAISS index.")

    parser.add_argument('--directory', type=str, required=True, help="Directory path containing the documents (PDF, CSV, JSON)")
    parser.add_argument('--output', type=str, required=True, help="Output directory to save the FAISS index")
    parser.add_argument('--model', type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)")

    args = parser.parse_args()

    directory_path = args.directory  # 文件夹路径
    output_dir = args.output  # 输出FAISS索引路径
    embedding_model_name = args.model  # 嵌入模型名称

    retriever = save_faiss_multi_vector_index(embedding_model_name, directory_path, output_dir)
    print(f"FAISS Index created and saved at {output_dir}!")