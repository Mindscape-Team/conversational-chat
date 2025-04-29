# Mental Health Support Chatbot

A context-aware mental health support chatbot that provides therapeutic responses based on user emotions and maintains conversation history.

## Features

- Emotion detection using state-of-the-art NLP models
- Context-aware responses
- Conversation memory
- Therapeutic techniques integration
- RESTful API interface

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

The server will start on `http://localhost:8000`

## API Endpoints

### Start a Session
- **Endpoint**: `POST /start_session`
- **Query Parameter**: `user_id`
- **Response**: 
```json
{
    "response": "Initial greeting message",
    "session_id": "unique_session_id"
}
```

### Send a Message
- **Endpoint**: `POST /send_message`
- **Body**:
```json
{
    "user_id": "user_id",
    "message": "Your message here"
}
```
- **Response**:
```json
{
    "response": "Chatbot's response",
    "session_id": "session_id"
}
```

### End a Session
- **Endpoint**: `POST /end_session`
- **Query Parameter**: `user_id`
- **Response**: Session summary with emotions, recommendations, and characteristics

### Get Session Summary
- **Endpoint**: `GET /session_summary/{session_id}`
- **Query Parameters**:
  - `include_summary` (boolean)
  - `include_recommendations` (boolean)
  - `include_emotions` (boolean)
  - `include_characteristics` (boolean)
  - `include_duration` (boolean)
  - `include_phase` (boolean)
- **Response**: Filtered session summary based on included parameters

### Health Check
- **Endpoint**: `GET /health`
- **Response**: `{"status": "healthy"}`

## Example Usage

Using curl:
```bash

curl -X POST "http://localhost:8000/start_session?user_id=user123"

# Send a message
curl -X POST "http://localhost:8000/send_message" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user123", "message": "I\'m feeling really anxious today"}'

# End a session
curl -X POST "http://localhost:8000/end_session?user_id=user123"

# Get session summary
curl "http://localhost:8000/session_summary/session_id_here"
```

## Important Notes

- This is not a replacement for professional mental health care
- Always seek professional help for serious mental health concerns


## Privacy and Security

- Conversations are stored in memory only
- No personal data is permanently stored
- The system is designed to be HIPAA-compliant
- Users are identified by unique IDs only
