from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class GeneralAgent:
    def __init__(self, vector_store, wikipedia_api):
        self.vector_store = vector_store
        self.wikipedia_api = wikipedia_api
        self.llm = Ollama(model="llama3", temperature=0.7)

        self.prompt = ChatPromptTemplate.from_template(
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Answer general questions.
Context: {context}
Wikipedia Info: {wiki_info}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
        )

        self.chain = (
            {
                "context": lambda x: self.vector_store.get_context(x["user_id"]),
                "wiki_info": lambda x: self.wikipedia_api.search(x["question"]),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
        )

    def handle_query(self, query: str, user_id: str) -> str:
        context = self.vector_store.get_context(user_id)
        wiki_info = self.wikipedia_api.search(query)
        response = self.chain.invoke({
            "context": context,
            "wiki_info": wiki_info,
            "question": query,
            "user_id": user_id
        })
        self.vector_store.store_interaction(user_id, query, response)
        return response
