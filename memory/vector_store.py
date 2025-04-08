from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

class VectorStore:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="llama3")
        self.db = FAISS.from_texts([""], self.embeddings)
        self.memory_log = {}  # { user_id: [{"role": "user/assistant", "content": "..."}, ...] }


    def store_interaction(self, user_id: str, query: str, response: str):
        """Stores a conversation turn in the vector database."""
        text = f"User ({user_id}): {query}\nBot: {response}"
        self.db.add_texts([text])
        # Also store structured memory
        if user_id not in self.memory_log:
            self.memory_log[user_id] = []
        self.memory_log[user_id].append({"role": "user", "content": query})
        self.memory_log[user_id].append({"role": "assistant", "content": response})


    def get_context(self, user_id: str, k=5) -> str:
        """Retrieves the last `k` interactions for context."""
        results = self.db.similarity_search(f"User ({user_id})", k=k)
        return "\n".join([doc.page_content for doc in results])