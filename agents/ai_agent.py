from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class AIAgent:
    def __init__(self, vector_store, wikipedia_api):
        self.vector_store = vector_store
        self.wikipedia_api = wikipedia_api

        self.llm = Ollama(
            model="llama3",
            temperature=0.5,
            top_p=0.7,
            repeat_penalty=1.1,
            num_ctx=2048
        )

        self.prompt = ChatPromptTemplate.from_template(
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI expert assistant. Provide detailed, technical answers about artificial intelligence.
Context: {context}
Wikipedia Info: {wiki_info}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
        )

        self.chain = (
            {
                "context": lambda x: x["context"],
                "wiki_info": lambda x: x["wiki_info"],
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
        )

    def _build_context(self, user_id):
        chat_log = self.vector_store.memory_log.get(user_id, [])
        return "\n".join([
            f"{entry['role'].capitalize()}: {entry['content']}"
            for entry in chat_log if entry['role'] in ("user", "assistant")
        ])

    def handle_query(self, query: str, user_id: str) -> str:
        context = self._build_context(user_id)
        wiki_info = self.wikipedia_api.search(query)

        response = self.chain.invoke({
            "context": context,
            "wiki_info": wiki_info,
            "question": query,
            "user_id": user_id
        })

        self.vector_store.store_interaction(user_id, query, response)
        return response

    def generate_candidates(self, query: str, user_id: str, n=1):
        context = self._build_context(user_id)
        wiki_info = self.wikipedia_api.search(query)

        responses = []
        for _ in range(n):
            response = self.chain.invoke({
                "context": context,
                "wiki_info": wiki_info,
                "question": query,
                "user_id": user_id
            })
            responses.append(response)
        return responses
