# RAGinBurgh
end-to-end RAG system for Pittsburgh and CMU related topics

## Enviroment
GPU resouce: AWS EC2 g5.2xlarge

```bash
conda activate pytorch
pip install -r requirements.txt
```

## runing inference
Notes: you may input a LANGCHAIN_API_KEY, this is only for LangSmith monitoring.
create a new bash file as the following in `RAGinBurgh\rag`

```bash
export LANGCHAIN_API_KEY="..."
query_file="path/to/query/file.csv"
output_file="path/to/output/file.csv"

python rag.py \
    --device cuda:0 \
    --create_bm25 \
    --generator "mistralai/Mistral-7B-Instruct-v0.2" \
    --faiss_output_dir "rag_faiss_index_10230232" \
    --normalize_embeddings \
    --compression_retriever \
    --few_shot \
    --query_file $query_file \
    --max_new_tokens 128 \
    --output_file $output_file \
    --directory_path "../data_collect/raw_documents" \
    --top_k 1
```

