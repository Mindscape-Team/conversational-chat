from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from chatbot import MentalHealthChatbot

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Chatbot",
    description="mental health support chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = MentalHealthChatbot(
    model_name=os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"),
    peft_model_path=os.getenv("PEFT_MODEL_PATH", "llama_fine_tuned"),
    therapy_guidelines_path=os.getenv("GUIDELINES_PATH", "guidelines.txt"),
    use_4bit=True
)

# Pydantic models for request/response
class MessageRequest(BaseModel):
    user_id: str
    message: str

class MessageResponse(BaseModel):
    response: str
    session_id: str

class SessionSummary(BaseModel):
    session_id: str
    user_id: str
    start_time: str
    end_time: str
    duration_minutes: float
    current_phase: str
    primary_emotions: list
    emotion_progression: list
    summary: str
    recommendations: list
    session_characteristics: dict

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Mental Health Chatbot API",
        "version": "1.0.0",
        "description": "API for mental health support chatbot",
        "endpoints": {
            "POST /start_session": "Start a new chat session",
            "POST /send_message": "Send a message to the chatbot",
            "POST /end_session": "End the current session",
            "GET /health": "Health check endpoint",
            "GET /docs": "API documentation (Swagger UI)",
            "GET /redoc": "API documentation (ReDoc)"
        }
    }

@app.post("/start_session", response_model=MessageResponse)
async def start_session(user_id: str):
    """Start a new chat session."""
    try:
        session_id, initial_message = chatbot.start_session(user_id)
        return MessageResponse(response=initial_message, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send_message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """Process a user message and return the chatbot's response."""
    try:
        response = chatbot.process_message(request.user_id, request.message)
        session = chatbot.conversations[request.user_id]
        return MessageResponse(response=response, session_id=session.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end_session", response_model=SessionSummary)
async def end_session(user_id: str):
    """End the current session and return the summary."""
    try:
        summary = chatbot.end_session(user_id)
        if not summary:
            raise HTTPException(status_code=404, detail="No active session found")
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 