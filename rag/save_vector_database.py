# docs with different types (json, pdf, csv) 
# input: raw file path;
# output: a retrieval

# step1: generate summaries for table; reference: https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb
# step2: retriever = MultiVectorRetriever
# step3: splitter

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf_loader import PDFLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

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

def split_document(document, chunk_size=500):
    text = document.page_content  # TODO?
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def save_faiss_multi_vector_index(embedding_model_name, source_files, file_types, output_dir):
    embedding_model_kwargs = {'device': 'cuda:0'}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}
    local_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )

    all_docs = []
    file_types = [sourse_file.lower().split('.')[-1] for sourse_file in source_files]
    for file_path, file_type in zip(source_files, file_types):
        if file_type not in ["pdf", "csv", "json"]:
            continue
        docs = load_documents(file_path, file_type)
        
        for doc in docs:
            chunks = split_document(doc) 
            for chunk in chunks:
                all_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    faiss_index = FAISS.from_documents(all_docs, local_embeddings)
    
    faiss_index.save_local(output_dir)

# 示例用法
source_files = ["/path/to/pdf1.pdf", "/path/to/data.csv", "/path/to/data.json"]
file_types = ["pdf", "csv", "json"]
save_faiss_multi_vector_index("sentence-transformers/all-MiniLM-L6-v2", source_files, file_types, "/output/faiss_index")