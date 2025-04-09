from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class AdmissionAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store

        self.llm = Ollama(
            model="llama3",
            temperature=0.5,
            top_p=0.5,
            repeat_penalty=1.1,
            num_ctx=2048
        )

        self.admission_data = {
            "deadlines": "2025-01-15",
            "requirements": "GPA 3.0+, CS prerequisites",
            "program_info": "4-year Computer Science program...",
            "general": "Please contact admissions@concordia.ca for general queries."
        }

        self.concordia_cs_data = {
            "program_name": "BCompSc - Bachelor of Computer Science",
            "duration": "4 years full-time",
            "deadlines": "Fall: March 1, Winter: November 1",
            "requirements": "High school diploma, math prerequisites, minimum GPA 3.0",
            "tuition": "Approx. CAD 7,500/year for Quebec residents, CAD 18,000/year for international students",
            "contact": "csadmissions@concordia.ca",
            "website": "https://www.concordia.ca/academics/undergraduate/computer-science.html"
        }

        self.prompt = ChatPromptTemplate.from_template(
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant specialized in Concordia University's Computer Science admissions. Provide clear, specific, and helpful answers.
Context: {context}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        )

        self.chain = (
            {
                "context": lambda x: x["context"],
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

    def handle_query(self, query, user_id):
        query_lower = query.lower()

        if "concordia" in query_lower and ("computer science" in query_lower or "cs" in query_lower):
            for key in self.concordia_cs_data:
                if key in query_lower:
                    return self.concordia_cs_data[key]

        for key in self.admission_data:
            if key in query_lower:
                return self.admission_data[key]

        context = self._build_context(user_id)
        response = self.chain.invoke({
            "question": query,
            "user_id": user_id,
            "context": context
        })

        self.vector_store.store_interaction(user_id, query, response)
        return response

    def generate_candidates(self, query: str, user_id: str, n=1):
        context = self._build_context(user_id)
        responses = []
        for _ in range(n):
            response = self.chain.invoke({
                "question": query,
                "user_id": user_id,
                "context": context
            })
            responses.append(response)
        return responses
