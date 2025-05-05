from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from chatbot import MentalHealthChatbot
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Chatbot",
    description="mental health support chatbot",
    version="1.0.0"
)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  
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

# pydantic models
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
    primary_emotions: List[str]
    emotion_progression: List[str]
    summary: str
    recommendations: List[str]
    session_characteristics: Dict[str, Any]

class UserReply(BaseModel):
    text: str
    timestamp: str
    session_id: str

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
    try:
        session_id, initial_message = chatbot.start_session(user_id)
        return MessageResponse(response=initial_message, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send_message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    try:
        response = chatbot.process_message(request.user_id, request.message)
        session = chatbot.conversations[request.user_id]
        return MessageResponse(response=response, session_id=session.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end_session", response_model=SessionSummary)
async def end_session(user_id: str):
    try:
        summary = chatbot.end_session(user_id)
        if not summary:
            raise HTTPException(status_code=404, detail="No active session found")
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/session_summary/{session_id}", response_model=SessionSummary)
async def get_session_summary(
    session_id: str,
    include_summary: bool = True,
    include_recommendations: bool = True,
    include_emotions: bool = True,
    include_characteristics: bool = True,
    include_duration: bool = True,
    include_phase: bool = True
):
    try:
        summary = chatbot.get_session_summary(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Session summary not found")
        
        filtered_summary = {
            "session_id": summary["session_id"],
            "user_id": summary["user_id"],
            "start_time": summary["start_time"],
            "end_time": summary["end_time"],
            "duration_minutes": summary.get("duration_minutes", 0.0),
            "current_phase": summary.get("current_phase", "unknown"),
            "primary_emotions": summary.get("primary_emotions", []),
            "emotion_progression": summary.get("emotion_progression", []),
            "summary": summary.get("summary", ""),
            "recommendations": summary.get("recommendations", []),
            "session_characteristics": summary.get("session_characteristics", {})
        }
        
        # Filter out fields based on include parameters
        if not include_summary:
            filtered_summary["summary"] = ""
        if not include_recommendations:
            filtered_summary["recommendations"] = []
        if not include_emotions:
            filtered_summary["primary_emotions"] = []
            filtered_summary["emotion_progression"] = []
        if not include_characteristics:
            filtered_summary["session_characteristics"] = {}
        if not include_duration:
            filtered_summary["duration_minutes"] = 0.0
        if not include_phase:
            filtered_summary["current_phase"] = "unknown"
        
        return filtered_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user_replies/{user_id}")
async def get_user_replies(user_id: str):
    try:
        replies = chatbot.get_user_replies(user_id)
        
        # Create a filename with user_id and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_replies_{user_id}_{timestamp}.json"
        filepath = os.path.join("user_replies", filename)
        
        # Ensure directory exists
        os.makedirs("user_replies", exist_ok=True)
        
        # Write replies to JSON file
        with open(filepath, 'w') as f:
            json.dump({
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "replies": replies
            }, f, indent=2)
        
        # Return the file
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 