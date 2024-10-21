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
import time
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
import faiss
from langsmith import traceable

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY']=""
os.environ['LANGCHAIN_PROJECT'] = "rag"

class PittsRAG():

    def __init__(self, generator, retrieval):
        # backbone model for RAG (generator)
        if type(generator) == str:
            # self.generator = HuggingFaceEndpoint(
            #     repo_id=generator,
            #     max_length=128,
            #     temperature=0.5,
            #     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            # )
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
        # 配置BitsAndBytes的設定，用於模型的量化以提高效率。
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,  # 啟用4位元量化
        #     bnb_4bit_compute_dtype=torch.float16,  # 計算時使用的數據類型
        #     bnb_4bit_quant_type="nf4",  # 量化類型
        #     bnb_4bit_use_double_quant=True,  # 使用雙重量化
        # )

        # 加載並配置模型，這裡使用了前面定義的量化配置。
        model_4bit = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",  # 自動選擇運行設備
        )

        # 加載模型的分詞器。
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # 創建一個用於文本生成的pipeline。
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model_4bit,
            tokenizer=tokenizer,
            use_cache=True,
            device_map="auto",
            max_new_tokens=256,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # 創建一個HuggingFacePipeline實例，用於後續的語言生成。
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

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

        return self.rag_chain.invoke(query)
    
    def batch_answer(self, query_file, output_file):
        if not os.path.exists(output_file):
          os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(query_file, "r") as f:
            queries = f.readlines()

        results = self.rag_chain.batch(queries)

        with open(output_file, "w") as f:
            for result in results:
                f.write(result + "\n")

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
    ## retriever
    parser.add_argument("--score_threshold", type=float, \
        default=0.3, help="Directory where the FAISS index is saved.")
    parser.add_argument("--k", type=int, \
        default=20, help="Number of documents to retrieve.")
    ## rag
    parser.add_argument("--generator", type=str, \
        default='SciPhi/SciPhi-Self-RAG-Mistral-7B-32k', help="The name of the generator model.")
    parser.add_argument("--query_file", type=str, \
        default='../QA/questions.txt', help="File containing the queries.")
    parser.add_argument("--output_file", type=str, \
        default='../model_output/answers.txt', help="File where the answers will be saved.")


    args = parser.parse_args()

    
    if args.create_faiss:
        retriever = save_faiss_multi_vector_index(args) \
            .as_retriever(search_type="similarity", search_kwargs={"score_threshold": args.score_threshold, "k": args.k})
    else:
        retriever = FAISS.load_local(args.faiss_output_dir, create_embedding(args), \
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, allow_dangerous_deserialization=True) \
                .as_retriever(search_type="similarity", search_kwargs={"score_threshold": args.score_threshold, "k": args.k})
    # else:
    # # Load the FAISS index from disk
    #     cpu_index = FAISS.load_local(args.faiss_output_dir, create_embedding(args), \
    #         distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, allow_dangerous_deserialization=True).index

    #     # 将 CPU 索引转移到 GPU 上
    #     res = faiss.StandardGpuResources()
    #     gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 0 是 GPU 设备编号

    #     # 创建 GPU 版的 FAISS 检索器
    #     retriever = FAISS(gpu_index, create_embedding(args)) \
    #         .as_retriever(search_type="similarity", search_kwargs={"score_threshold": args.score_threshold, "k": args.k})

    rag = PittsRAG(generator=args.generator, retrieval=retriever)
    rag.batch_answer(args.query_file, args.output_file)
    # result = rag.inference("When is Yalda Night held?")
    # print(result)

    # rag.save_answer(args.query_file, args.output_file)


if __name__ == '__main__':
    main()
