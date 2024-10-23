# docs with different types (json, pdf, csv) 
# input: raw file path;
# output: a retrieval

# TODO ####################################################################################
# step1: generate summaries for table; reference: https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb
# step2: retriever = MultiVectorRetriever
# step3: splitter
# step4: save_faiss_multi_vector_index
###########################################################################################
import re
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
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever

# def delete_special_chars(text):
#     return ''.join(e for e in text if e.isalnum())

# def json_to_markdown_documents(json_file_path):
#     documents = []
    
#     def json_to_markdown_string(json_data, level=1):
#         markdown_string = ""
#         for key, value in json_data.items():
#             markdown_string += f"{'#' * level} {key}\n"
#             if isinstance(value, dict):
#                 markdown_string += json_to_markdown_string(value, level + 1)
#             elif isinstance(value, list):
#                 for item in value:
#                     markdown_string += f"- {item}\n"

#             else:
#                 markdown_string += f"{value}\n"
#         return markdown_string

# def json_list_to_markdown(data):
#         markdown_lines = []
        
#         if isinstance(data, list):
#             for item in data:
#                 markdown_lines.extend(json_list_to_markdown(item))
#         else:
#             markdown_lines.append(f"{str(data)}")

#         return markdown_lines

#     with open(json_file_path, 'r') as json_file:
#         json_data = json.load(json_file)
#         # Loop through each key and value to create separate Documents
#         if isinstance(json_data, list):
#             try:
#                 markdown_content = ''
#                 for json_item in json_data:
#                     if isinstance(json_item, dict):
#                         markdown_content += json_to_markdown_string(json_item)+'\n'
#                     else:
#                         documents.append(Document(page_content =  '\n'.join(json_list_to_markdown(json_data))))
#                 assert isinstance(markdown_content, str)
#                 documents.append(Document(page_content =  markdown_content))
#             except:
#                 documents.append(Document(page_content =  '\n'.join(json_list_to_markdown(json_data))))

#         elif isinstance(json_data, dict):
#             markdown_content = json_to_markdown_string(json_data)
#             assert isinstance(markdown_content, str)
#             documents.append(Document(page_content =  markdown_content))
        
#     return documents

def delete_special_chars(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_repeated_chars(text, repeat_count=6):
    return re.sub(r'(.)\1{' + str(repeat_count - 1) + r',}', '', text)

def flatten_json(json_data, parent_key='', sep='_'):
    items = []
    for key, value in json_data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_json(value, new_key, sep=sep).items())
        elif isinstance(value, list):
            for i, item in enumerate(value):
                items.extend(flatten_json({f"{new_key}_{i}": item}).items())
        else:
            items.append((new_key, value))
    return dict(items)

def json_to_markdown_documents(json_file_path):
    documents = []
    
    def json_to_markdown_string(json_data):
        markdown_string = ""
        for key, value in json_data.items():
            if isinstance(value, dict):
                markdown_string += key + ": "
                markdown_string += json_to_markdown_string(value)
            elif isinstance(value, list):
                for item in value:
                    if len(item) <= 1:
                        continue
                    markdown_string += f"{delete_special_chars(str(item))}. "
            else:
                clean_value = delete_special_chars(str(value))
                clean_value = remove_repeated_chars(clean_value)
                markdown_string += f"{key}: {clean_value}. "
            markdown_string += "\n"
        return markdown_string

    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

        if isinstance(json_data, list):
            for json_item in json_data:
                if isinstance(json_item, dict):
                    markdown_content = json_to_markdown_string(json_item)
                    documents.append(Document(page_content=markdown_content))
                else:
                    documents.append(Document(page_content=str(json_item)))
        elif isinstance(json_data, dict):
            markdown_content = json_to_markdown_string(json_data)
            documents.append(Document(page_content=markdown_content))
        
    return documents

def load_documents(file_path, file_type):
    # if file_type == "pdf":
    #     loader = PyPDFLoader(file_path)
    if file_type == "csv":
        loader = CSVLoader(file_path)
    elif file_type == "json":
        # 转markdown -> 返回List[Document] 
        return json_to_markdown_documents(file_path) # return: List[Document]
    elif file_type == "txt":
        with open(file_path, 'r') as f:
            text = f.read()
        return [Document(page_content=text)]
    
    return loader.load() # return: List[Document]

def create_embedding(args):
    embedding_model_kwargs = {'device': args.device}
    embedding_encode_kwargs = {
        'batch_size': args.embedding_batch_size,
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
    dict_list = []
    # BUILD DOCUMENTS ----
    if isinstance(args.directory_path, str):
        dict_list = [args.directory_path]
    elif isinstance(args.directory_path, list):
        dict_list = args.directory_path

    for directory in dict_list:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_type = file.split('.')[-1].lower()
                if file_type not in ["pdf", "csv", "json"]:
                    continue
                try:
                    docs = load_documents(file_path, file_type) #  List[Document]
                    if sum([len(doc.page_content) for doc in docs]) < args.chunk_size:
                        all_docs.extend(docs)
                    else:
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

    if args.create_bm25:
        save_bm25_retriever(all_docs, args.bm25_save_path, args.k)

    return faiss_retriever

def save_bm25_retriever(docs, bm25_save_path, k):
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k
    bm25_retriever.save_local("rag_bm25_index")
    return bm25_retriever

