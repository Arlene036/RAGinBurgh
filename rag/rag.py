from prompt import *
from save_vector_database import *
import argparse
import os

class PittsRAG():

    def __init__(self, generator, retrieval):
        # backbone model for RAG (generator)
        if type(generator) == str:
            self.generator = HuggingFaceEndpoint(
                repo_id=generator,
                max_length=128,
                temperature=0.5,
                huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            )
        else:
            self.generator = generator

        self.retrieval = retrieval # retriever for RAG
        self.rag_prompt = RAG_PROMPT

    def compute_loss(self, outputs, labels):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs, labels)

    def train(self, train_data, train_labels, batch_size=32, epochs=3, lr=1e-5):
        self.optimizer = optim.Adam(list(self.generator.parameters()) + list(self.retriever.parameters()), lr=lr)
        
        train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.generator.train()
            self.retriever.train()
            
            total_loss = 0.0
            for batch_data, batch_labels in train_loader:
                context = self.retriever(batch_data) 

                outputs = self.generator(context, batch_data) # ??????????????
                
                loss = self.compute_loss(outputs, batch_labels)  
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

        torch.save(self.generator.state_dict(), "generator_model.pth")
        torch.save(self.retriever.state_dict(), "retriever_model.pth")
        print("Training complete and models saved.")


    def eval(self, eval_data, eval_labels, batch_size=32):
        eval_loader = DataLoader(list(zip(eval_data, eval_labels)), batch_size=batch_size)

        self.generator.eval()
        self.retriever.eval()
        
        total_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in eval_loader:
                context = self.retriever(batch_data)

                outputs = self.generator(context, batch_data)

                loss = self.compute_loss(outputs, batch_labels)
                total_loss += loss.item()

        print(f"Evaluation Loss: {total_loss:.4f}")

    def inference(self, query):
        # TODO: query rewriting (HyDE) OR fine-tuning
        prompt = PromptTemplate.from_template(self.rag_prompt)

        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.generator
            | StrOutputParser()
        )

        return rag_chain(query)
    
    def save_answer(self, query_file, output_file):
        with open(query_file, "r") as f:
            queries = f.readlines()

        with open(output_file, "w") as f:
            for query in queries:
                answer = self.inference(query)
                f.write(answer + "\n")

def main():
    parser = argparse.ArgumentParser()
    ## faiss
    parser.add_argument("--embedding_model_name", type=str, \
        default="sentence-transformers/multi-qa-mpnet-base-dot-v1", help="The name of the embedding model.")
    parser.add_argument("--directory_path", type=str, \
        default='../data_collect', help="Path to the directory containing documents.")
    parser.add_argument("--output_dir", type=str, \
        default='rag_faiss_index', help="Directory where the FAISS index will be saved.")
    parser.add_argument("--device", type=str, \
        default="cuda:0", help="Device to run the embedding model on, e.g., 'cpu' or 'cuda:0'.")
    parser.add_argument("--batch_size", type=int, \
        default=32, help="Batch size for encoding embeddings.")
    parser.add_argument("--normalize_embeddings", \
        action="store_true", help="Whether to normalize embeddings.")
    parser.add_argument("--show_progress_bar", \
        action="store_true", help="Whether to show a progress bar during encoding.")
    parser.add_argument("--chunk_size", type=int, \
        default=1000, help="Chunk size for splitting documents.")
    parser.add_argument("--chunk_overlap", type=int, \
        default=200, help="Overlap between chunks when splitting documents.")
    
    ## rag
    parser.add_argument("--generator", type=str, \
        default='SciPhi/SciPhi-Self-RAG-Mistral-7B-32k', help="The name of the generator model.")
    parser.add_argument("--query_file", type=str, \
        default='../QA/queries.txt', help="File containing the queries.")
    parser.add_argument("--output_file", type=str, \
        default='../model_output/answers.txt', help="File where the answers will be saved.")

    args = parser.parse_args()


    retriever = save_faiss_multi_vector_index(args)
    #     embedding_model_name=args.embedding_model_name,
    #     directory_path=args.directory_path,
    #     output_dir=args.output_dir,
    #     device=args.device,
    #     batch_size=args.batch_size,
    #     normalize_embeddings=args.normalize_embeddings,
    #     show_progress_bar=args.show_progress_bar,
    #     chunk_size=args.chunk_size,
    #     chunk_overlap=args.chunk_overlap
    # )

    rag = PittsRAG(generator=args.generator, retrieval=retriever)



if __name__ == '__main__':
    main()
