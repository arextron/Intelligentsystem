from fastapi import FastAPI, HTTPException
from agents.general_agent import GeneralAgent
from agents.admission_agent import AdmissionAgent
from agents.ai_agent import AIAgent
from memory.vector_store import VectorStore
from utils.external_api import WikipediaAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import difflib
import csv
import os
from typing import Optional

from utils.reward_model import RewardModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Initialization ===
try:
    vector_store = VectorStore()
    wikipedia_api = WikipediaAPI()
    general_agent = GeneralAgent(vector_store, wikipedia_api)
    admission_agent = AdmissionAgent(vector_store)
    ai_agent = AIAgent(vector_store, wikipedia_api)
    reward_model = RewardModel()
    reward_model.load()
except Exception as e:
    print(f"Initialization error: {e}")
    raise

# === Models ===
class ChatRequest(BaseModel):
    user_input: str
    user_id: str

class Feedback(BaseModel):
    user_id: str
    input: str
    response: str
    agent: str
    contextual: bool
    rating: int  # 1 to 5

# === Intent Detection ===
AI_KEYWORDS = {"ai", "ml", "machine learning", "deep learning", "dl", "nlp", "chatbot", "transformer","natural language processing", "artificial intelligence", "model", "algorithm", "data", "training", "inference","research", "paper", "study", "experiment", "results", "evaluation", "accuracy", "performance", "benchmark", "dataset", "feature", "label", "training set", "test set", "validation set", "hyperparameter", "tuning"}
ADMISSION_KEYWORDS = {"admission", "concordia", "computer science", "cs", "apply", "gpa", "deadline"," requirements", "tuition", "program", "application", "acceptance", "status", "documents", "transcripts", "english proficiency", "ielts", "toefl", "gre", "sat", "act", "interview", "offer letter"," acceptance letter", "deferral", "transfer", "international student", "visa", "scholarship", "financial aid", "tuition fee", "cost of living", "housing", "accommodation", "campus life", "student services", "orientation", "registration", "enrollment", "course load", "academic calendar"}

def get_intent_agent(user_input: str):
    tokens = set(user_input.lower().split())
    if tokens & AI_KEYWORDS:
        return "ai"
    if tokens & ADMISSION_KEYWORDS:
        return "admission"
    return "general"

def is_followup(current_query: str, last_query: str, threshold: float = 0.2) -> bool:
    if not last_query:
        return False
    ratio = difflib.SequenceMatcher(None, current_query.lower(), last_query.lower()).ratio()
    return ratio > threshold or any(word in current_query.lower() for word in {"what", "and", "also", "about", "more", "that"})

# === Chat Endpoint ===
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_id = request.user_id
        user_input = request.user_input.strip().lower()

        greetings = {"hi", "hello", "hey"}
        farewells = {"bye", "goodbye", "see you"}

        if user_input in greetings:
            return {
                "response": "Hi there! 👋 How can I assist you today?",
                "agent": "general",
                "contextual": False
            }

        if user_input in farewells:
            return {
                "response": "Goodbye! 👋 Have a great day!",
                "agent": "general",
                "contextual": False
            }

        history = vector_store.memory_log.get(user_id, [])
        last_query = ""
        for h in reversed(history):
            if h["role"] == "user":
                last_query = h["content"]
                break

        is_contextual = is_followup(user_input, last_query)
        agent_type = get_intent_agent(user_input)

        if agent_type == "admission":
            candidates = admission_agent.generate_candidates(user_input, user_id, n=1)
        elif agent_type == "ai":
            candidates = ai_agent.generate_candidates(user_input, user_id, n=1)
        else:
            candidates = general_agent.generate_candidates(user_input, user_id, n=1)

        if hasattr(reward_model, 'model'):
            response = max(candidates, key=lambda r: reward_model.predict(user_input, r))
        else:
            response = candidates[0]
        
        vector_store.store_interaction(user_id, user_input, response)

        return {
            "response": response,
            "agent": agent_type,
            "contextual": is_contextual,
            "feedback_prompt": "How would you rate this response? (1–5)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Feedback Endpoint ===
@app.post("/feedback")
async def collect_feedback(feedback: Feedback):
    log_file = "chat_logs.csv"
    is_new_file = not os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp", "user_id", "input", "response", "agent", "contextual", "rating"])
        writer.writerow([
            datetime.now(),
            feedback.user_id,
            feedback.input,
            feedback.response,
            feedback.agent,
            feedback.contextual,
            feedback.rating
        ])
    return {"status": "Feedback logged"}
