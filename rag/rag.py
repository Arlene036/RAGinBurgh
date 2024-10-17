from prompt import *


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


