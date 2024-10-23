from prompt import *
from save_vector_database import *
import argparse
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFacePipeline
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
import time
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import faiss
from sentence_transformers import SentenceTransformer, util
from langsmith import traceable
import pandas as pd
from pydantic import BaseModel, Field

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY']=""
os.environ['LANGCHAIN_PROJECT'] = "ragAWS"

class RAGReranker(BaseDocumentCompressor):
    """Custom document compressor based on a query."""
    def compress_documents(
        self,
        documents, # Sequence[Document]
        query: str,
        callbacks=None
    ): # -> Sequence[Document]
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        query_embedding = model.encode(query, convert_to_tensor=True)
        doc_text = [doc.page_content for doc in documents]
        doc_embeddings = model.encode(doc_text, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)
        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs]

class PittsRAG():

    def __init__(self, generator, retrieval, generator_batch_size=10, max_new_tokens=128, tok_k=5):
        # backbone model for RAG (generator)
        self.generator_batch_size = generator_batch_size
        self.max_new_tokens = max_new_tokens
        self.tok_k = tok_k
        if type(generator) == str:
            self.generator = self.get_generator(generator)
        else:
            self.generator = generator

        self.retrieval = retrieval # retriever for RAG
        self.rag_prompt = RAG_PROMPT

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = PromptTemplate.from_template(self.rag_prompt)

        self.rag_chain = (
            {"context": self.retrieval | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.generator
            | StrOutputParser()
        )
     

    def get_generator(self, model_id):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  
            bnb_4bit_compute_dtype=torch.float16,  
            bnb_4bit_quant_type="nf4",  
            bnb_4bit_use_double_quant=True,  
        )

        model_4bit = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",  
            quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        text_generation_pipeline = pipeline(
            "text-generation",
            model=model_4bit,
            tokenizer=tokenizer,
            use_cache=True,
            device_map="auto",
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_k=self.tok_k,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline, model_id = model_id, batch_size = self.generator_batch_size)

        return llm
    # def compute_loss(self, outputs, labels):
    #     criterion = torch.nn.CrossEntropyLoss()
    #     return criterion(outputs, labels)

    # def train(self, train_data, train_labels, batch_size=32, epochs=3, lr=1e-5):
    #     self.optimizer = optim.Adam(list(self.generator.parameters()) + list(self.retriever.parameters()), lr=lr)
        
    #     train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=batch_size, shuffle=True)

    #     for epoch in range(epochs):
    #         self.generator.train()
    #         self.retriever.train()
            
    #         total_loss = 0.0
    #         for batch_data, batch_labels in train_loader:
    #             context = self.retriever(batch_data) 

    #             outputs = self.generator(context, batch_data) # ??????????????
                
    #             loss = self.compute_loss(outputs, batch_labels)  
    #             total_loss += loss.item()

    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    #         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

    #     torch.save(self.generator.state_dict(), "generator_model.pth")
    #     torch.save(self.retriever.state_dict(), "retriever_model.pth")
    #     print("Training complete and models saved.")


    # def eval(self, eval_data, eval_labels, batch_size=32):
    #     eval_loader = DataLoader(list(zip(eval_data, eval_labels)), batch_size=batch_size)

    #     self.generator.eval()
    #     self.retriever.eval()
        
    #     total_loss = 0.0
    #     with torch.no_grad():
    #         for batch_data, batch_labels in eval_loader:
    #             context = self.retriever(batch_data)

    #             outputs = self.generator(context, batch_data)

    #             loss = self.compute_loss(outputs, batch_labels)
    #             total_loss += loss.item()

    #     print(f"Evaluation Loss: {total_loss:.4f}")

    @traceable
    def inference(self, query):
        # TODO: query rewriting (HyDE) OR fine-tuning

        result = self.rag_chain.invoke(query)
        if query in result:
            result = result[len(query):].strip()  # 去除 prompt 部分

        return result
    

    def batch_answer(self, query_file, output_file):
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(query_file, "r") as f:
            queries = f.readlines()

        results = self.rag_chain.batch(queries)

        data = []

        for query, result in zip(queries, results):
            answer = result.split('Helpful Answer: ')[-1].strip()
            data.append([query.strip(), answer])

        df = pd.DataFrame(data, columns=["query", "answer"])
        df.to_csv(output_file, index=False)

def get_llm(model_id):
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_new_tokens=128,
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline, model_id = model_id)

    return llm

def main():
    parser = argparse.ArgumentParser()
    ## faiss
    parser.add_argument("--embedding_model_name", type=str, \
        default="sentence-transformers/multi-qa-mpnet-base-dot-v1", help="The name of the embedding model.")
    parser.add_argument("--directory_path", type=str, \
        default='../data_collect', help="Path to the directory containing documents.")
    parser.add_argument("--faiss_output_dir", type=str, \
        default='rag_faiss_index', help="Directory where the FAISS index will be saved.")
    parser.add_argument("--device", type=str, \
        default="cuda:0", help="Device to run the embedding model on, e.g., 'cpu' or 'cuda:0'.")
    parser.add_argument("--embedding_batch_size", type=int, \
        default=32, help="Batch size for encoding embeddings.")
    parser.add_argument("--normalize_embeddings", \
        action="store_true", help="Whether to normalize embeddings.")
    parser.add_argument("--show_progress_bar", \
        action="store_true", help="Whether to show a progress bar during encoding.")
    parser.add_argument("--chunk_size", type=int, \
        default=1000, help="Chunk size for splitting documents.")
    parser.add_argument("--chunk_overlap", type=int, \
        default=200, help="Overlap between chunks when splitting documents.")
    parser.add_argument("--create_faiss", \
        action="store_true", help="Whether to create a new FAISS index.")
    ## bm25
    parser.add_argument("--create_bm25", \
        action="store_true", help="Whether to create a new BM25 index.")
    parser.add_argument("--bm25_save_path", type=str, \
        default='rag_bm25_index', help="Directory where the BM25 index will be saved.")
    ## retriever
    parser.add_argument("--score_threshold", type=float, \
        default=0.3, help="Directory where the FAISS index is saved.")
    parser.add_argument("--k", type=int, \
        default=20, help="Number of documents to retrieve.")
    parser.add_argument("--ensemble_retriever", \
        action="store_true", help="Whether to use ensemble retriever.")
    parser.add_argument("--compression_retriever", \
        action="store_true", help="Whether to use compression retriever.")
    ## reranker
    parser.add_argument("--reranker_model", type=str, \
        default='sentence-transformers/all-MiniLM-L6-v2', help="rerank")
    ## rag
    parser.add_argument("--generator", type=str, \
        default='SciPhi/SciPhi-Self-RAG-Mistral-7B-32k', help="The name of the generator model.")
    parser.add_argument("--query_file", type=str, \
        default='../QA/questions.txt', help="File containing the queries.")
    parser.add_argument("--output_file", type=str, \
        default='../model_output/answers.csv', help="File where the answers will be saved.")
    parser.add_argument("--max_new_tokens", type=int, \
        default=128)
    parser.add_argument("--generator_batch_size", type=int, \
        default=10)
    parser.add_argument("--top_k", type=int, \
        default=5)

    args = parser.parse_args()

    if args.create_faiss:
        faiss_retriever = save_faiss_multi_vector_index(args) \
            .as_retriever(search_type="similarity", search_kwargs={"score_threshold": args.score_threshold, "k": args.k})
    else:
        faiss_retriever = FAISS.load_local(args.faiss_output_dir, create_embedding(args), \
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, allow_dangerous_deserialization=True) \
                .as_retriever(search_type="similarity", search_kwargs={"score_threshold": args.score_threshold, "k": args.k})
        if args.ensemble_retriever:
            bm25_retriever = BM25Retriever.load_local(args.bm25_save_path)
            # ADD Ensemble Retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
            )
            
    ensemble_retriever = None
    if ensemble_retriever is not None:
        retriever = ensemble_retriever
    else:
        retriever = faiss_retriever
    
    if args.compression_retriever:
        # ADD Compression Retriever
        # TODO
        compression_retriever = ContextualCompressionRetriever( # BaseRetriever
            base_compressor=RAGReranker(), # BaseDocumentCompressor
            base_retriever=retriever
        )

        retriever = compression_retriever

        
    rag = PittsRAG(generator=args.generator, retrieval=retriever, max_new_tokens = args.max_new_tokens, 
                    generator_batch_size = args.generator_batch_size, tok_k=args.top_k)
    # rag.batch_answer(args.query_file, args.output_file)
    result = rag.inference("When is Yalda Night held?")
    print(result)

    # rag.save_answer(args.query_file, args.output_file)


if __name__ == '__main__':
    main()
