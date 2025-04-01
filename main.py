# main.py
from fastapi import FastAPI
from agents.general_agent import GeneralAgent
from agents.admission_agent import AdmissionAgent
from agents.ai_agent import AIAgent
from memory.vector_store import VectorStore
from utils.external_api import WikipediaAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change in production)
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize components
try:
    vector_store = VectorStore()
    wikipedia_api = WikipediaAPI()
    general_agent = GeneralAgent(vector_store, wikipedia_api)
    admission_agent = AdmissionAgent(vector_store)
    ai_agent = AIAgent(vector_store, wikipedia_api)
except Exception as e:
    print(f"Initialization error: {e}")
    raise

# Define request model
class ChatRequest(BaseModel):
    user_input: str
    user_id: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        agent_type = "general"  # Default
        if "admission" in request.user_input.lower():
            response = admission_agent.handle_query(request.user_input, request.user_id)
            agent_type = "admission"
        elif "ai" in request.user_input.lower():
            response = ai_agent.handle_query(request.user_input, request.user_id)
            agent_type = "ai"
        else:
            response = general_agent.handle_query(request.user_input, request.user_id)
        
        return {
            "response": response,
            "agent": agent_type  # Add agent info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))