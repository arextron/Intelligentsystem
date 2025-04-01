# agents/ai_agent.py
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class AIAgent:
    def __init__(self, vector_store, wikipedia_api):
        self.vector_store = vector_store
        self.wikipedia_api = wikipedia_api
        
        # Initialize Ollama with proper parameters
        self.llm = Ollama(
            model="llama3",
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=4096
        )
        
        # Set up the prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an AI expert assistant. Provide detailed, technical answers about artificial intelligence.
            Context: {context}
            Wikipedia Info: {wiki_info}<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Question: {question}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>"""
        )
        
        # Create the processing chain
        self.chain = (
            {
                "context": lambda x: self.vector_store.get_context(x["question"]),
                "wiki_info": lambda x: self.wikipedia_api.search(x["question"]),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
        )
        
    def handle_query(self, query, context):
        try:
            return self.chain.invoke({"question": query})
        except Exception as e:
            return f"Error processing your query: {str(e)}"