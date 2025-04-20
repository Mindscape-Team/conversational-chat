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
python mental_health_chatbot.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### Chat with the Bot
- **Endpoint**: `POST /chat/{user_id}`
- **Body**:
```json
{
    "text": "Your message here"
}
```

### Get Conversation History
- **Endpoint**: `GET /history/{user_id}`

## Example Usage

Using curl:
```bash
# Send a message
curl -X POST "http://localhost:8000/chat/user123" \
     -H "Content-Type: application/json" \
     -d '{"text": "I\'m feeling really anxious today"}'

# Get conversation history
curl "http://localhost:8000/history/user123"
```

## Important Notes

- This chatbot is not a replacement for professional mental health care
- Always seek professional help for serious mental health concerns
- The chatbot is designed to provide support and coping strategies
- All conversations are stored in memory and will be lost when the server restarts

## Privacy and Security

- Conversations are stored in memory only
- No personal data is permanently stored
- The system is designed to be HIPAA-compliant
- Users are identified by unique IDs only
