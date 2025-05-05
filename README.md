# Mental Health Support Chatbot

A context-aware mental health support chatbot that provides therapeutic responses based on user emotions and maintains conversation history.

## Features

- Emotion detection using state-of-the-art NLP models
- Context-aware responses
- Conversation memory
- Therapeutic techniques integration
- Risk flag detection and crisis intervention
  - Automatic detection of high-risk messages
  - Immediate crisis response protocol
  - Professional support referral system
  - Emergency contact information
- RESTful API interface
- Session management and summaries
- User reply tracking for another depression and anxiety detection from text.


## Risk Flag Detection

The chatbot automatically monitors messages for potential risk indicators and provides appropriate crisis intervention responses.

### Risk Indicators
The system detects various risk-related keywords and phrases, including but not limited to:
- Self-harm references
- Suicidal ideation
- Extreme emotional distress
- Crisis situations

### Crisis Response Protocol
When risk flags are detected:
1. Immediate crisis response is triggered
2. User is provided with:
   - Emergency contact information
   - Professional support options
   - Immediate coping strategies
3. Option to connect with licensed professionals
4. Grounding exercises and calming techniques

### Example Crisis Response
```json
{
    "response": """ I'm really sorry you're feeling this way — it sounds incredibly heavy,
and I want you to know that you're not alone. You don't have to face this by yourself. Our app has licensed mental health professionals who are ready to support you. I can connect you right now if you'd like. In the meantime,
I'm here to listen and talk with you. You can also do grounding exercises or calming techniques with me if you prefer.
 Would you like to connect with a professional now, or would you prefer to keep talking with me for a bit? Either way, I'm here for you.""" ,
    "session_id": "user123_20240314103000",
    "risk_detected": true,
    "crisis_protocol_activated": true
}
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the required NLTK data:
```bash
python -m nltk.downloader punkt
```

3. Run the chatbot server:
```bash
python app.py
```

The server will start on `http://127.0.0.1:8000`

## API Documentation

### Base URL
```
http://127.0.0.1:8000
```

### API Endpoints

#### 1. Start a Session
```http
POST /start_session?user_id={user_id}
```

Example:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/start_session?user_id=user123' \
  -H 'accept: application/json'
```

Response:
```json
{
    "response": "Hello! I'm here to support you today. How have you been feeling lately?",
    "session_id": "user123_20240314103000"
}
```

#### 2. Send a Message
```http
POST /send_message
Content-Type: application/json

{
    "user_id": "user123",
    "message": "I'm feeling anxious today"
}
```

Example:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/send_message' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "user123",
    "message": "I'\''m feeling anxious today"
  }'
```

Response:
```json
{
    "response": "I understand you're feeling anxious. Can you tell me more about what's causing this?",
    "session_id": "user123_20240314103000"
}
```

#### 3. Get User Replies
```http
GET /user_replies/{user_id}
```

Example:
```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/user_replies/user123' \
  -H 'accept: application/json'
```

Response:
```json
{
    "user_id": "user123",
    "timestamp": "2024-03-14T10:30:00",
    "replies": [
        {
            "text": "I'm feeling anxious today",
            "timestamp": "2024-03-14T10:30:00",
            "session_id": "user123_20240314103000"
        }
    ]
}
```

#### 4. Get Session Summary
```http
GET /session_summary/{session_id}?include_summary={boolean}&include_recommendations={boolean}&include_emotions={boolean}&include_characteristics={boolean}&include_duration={boolean}&include_phase={boolean}
```

Example:
```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/session_summary/user123_20240314103000?include_summary=true&include_recommendations=true&include_emotions=true&include_characteristics=false&include_duration=false&include_phase=false' \
  -H 'accept: application/json'
```

Response:
```json
{
    "session_id": "user123_20240314103000",
    "user_id": "user123",
    "start_time": "2024-03-14T10:30:00",
    "end_time": "2024-03-14T10:45:00",
    "summary": "Session focused on anxiety management...",
    "recommendations": [
        "Practice deep breathing exercises",
        "Consider journaling your thoughts"
    ],
    "primary_emotions": ["anxiety", "stress"],
    "emotion_progression": ["anxiety", "calm"],
    "duration_minutes": 0.0,
    "current_phase": "unknown",
    "session_characteristics": {}
}
```

#### 5. End Session
```http
POST /end_session?user_id={user_id}
```

Example:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/end_session?user_id=user123' \
  -H 'accept: application/json'
```

Response: Complete session summary with all fields.

#### 6. Health Check
```http
GET /health
```

Example:
```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/health' \
  -H 'accept: application/json'
```

Response:
```json
{
    "status": "healthy"
}
```

## Integration Guidelines

### Best Practices
1. Always store the `session_id` returned from `/start_session`
2. Use the same `user_id` throughout a conversation
3. Include appropriate error handling for API responses
4. Monitor the health endpoint for system status

### Error Handling
The API returns standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

Error responses include a detail message:
```json
{
    "detail": "Error message here"
}
```


## Important Notes

- This is not a replacement for professional mental health care
- Always seek professional help for serious mental health concerns


## Privacy and Security

- Conversations are stored in memory only
- No personal data is permanently stored
- The system is designed to be HIPAA-compliant
- Users are identified by unique IDs only
