import os
import logging
import json
import torch
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

# Model imports
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS

# Import FlowManager
from conversation_flow import FlowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("mental_health_chatbot.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set environment variables
os.environ.update({
    'TRANSFORMERS_VERBOSITY': 'error',
    'TOKENIZERS_PARALLELISM': 'false',
    'BITSANDBYTES_NOWELCOME': '1'
})

# Define base directory and paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")
SESSION_DATA_PATH = os.path.join(BASE_DIR, "session_data")
SUMMARIES_DIR = os.path.join(BASE_DIR, "session_summaries")

# Create necessary directories
for directory in [MODELS_DIR, VECTOR_DB_PATH, SESSION_DATA_PATH, SUMMARIES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Pydantic models
class Message(BaseModel):
    text: str = Field(..., description="The content of the message")
    timestamp: str = Field(None, description="ISO format timestamp of the message")
    role: str = Field("user", description="The role of the message sender (user or assistant)")

class SessionSummary(BaseModel):
    session_id: str = Field(
        ...,
        description="Unique identifier for the session",
        examples=["user_789_session_20240314"]
    )
    user_id: str = Field(
        ...,
        description="Identifier of the user",
        examples=["user_123"]
    )
    start_time: str = Field(
        ...,
        description="ISO format start time of the session"
    )
    end_time: str = Field(
        ...,
        description="ISO format end time of the session"
    )
    message_count: int = Field(
        ...,
        description="Total number of messages in the session"
    )
    duration_minutes: float = Field(
        ...,
        description="Duration of the session in minutes"
    )
    primary_emotions: List[str] = Field(
        ...,
        min_items=1,
        description="List of primary emotions detected",
        examples=[
            ["anxiety", "stress"],
            ["joy", "excitement"],
            ["sadness", "loneliness"]
        ]
    )
    emotion_progression: List[Dict[str, float]] = Field(
        ...,
        description="Progression of emotions throughout the session",
        examples=[
            [
                {"anxiety": 0.8, "stress": 0.6},
                {"calm": 0.7, "anxiety": 0.3},
                {"joy": 0.9, "calm": 0.8}
            ]
        ]
    )
    summary_text: str = Field(
        ...,
        description="Text summary of the session",
        examples=[
            "The session focused on managing work-related stress and developing coping strategies. The client showed improvement in recognizing stress triggers and implementing relaxation techniques.",
            "Discussion centered around relationship challenges and self-esteem issues. The client expressed willingness to try new communication strategies."
        ]
    )
    recommendations: Optional[List[str]] = Field(
        None,
        description="Optional recommendations based on the session"
    )

class Conversation(BaseModel):
    user_id: str = Field(
        ...,
        description="Identifier of the user",
        examples=["user_123"]
    )
    session_id: str = Field(
        "",
        description="Identifier of the current session"
    )
    start_time: str = Field(
        "",
        description="ISO format start time of the conversation"
    )
    messages: List[Message] = Field(
        [],
        description="List of messages in the conversation",
        examples=[
            [
                Message(text="I'm feeling anxious", role="user"),
                Message(text="I understand you're feeling anxious. Can you tell me more about what's causing this?", role="assistant")
            ]
        ]
    )
    emotion_history: List[Dict[str, float]] = Field(
        [],
        description="History of emotions detected",
        examples=[
            [
                {"anxiety": 0.8, "stress": 0.6},
                {"calm": 0.7, "anxiety": 0.3}
            ]
        ]
    )
    context: Dict[str, Any] = Field(
        {},
        description="Additional context for the conversation",
        examples=[
            {
                "last_emotion": "anxiety",
                "conversation_topic": "work stress",
                "previous_sessions": 3
            }
        ]
    )
    is_active: bool = Field(
        True,
        description="Whether the conversation is currently active",
        examples=[True, False]
    )

class MentalHealthChatbot:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        peft_model_path: str = "nada013/mental-health-chatbot", 
        therapy_guidelines_path: str = None,
        use_4bit: bool = True,
        device: str = None
    ):
        # Set device (cuda if available, otherwise cpu)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.peft_model_path = peft_model_path
        
        # Initialize emotion detection model
        logger.info("Loading emotion detection model")
        self.emotion_classifier = self._load_emotion_model()
        
        # Initialize LLAMA model
        logger.info(f"Loading LLAMA model: {model_name}")
        self.llama_model, self.llama_tokenizer, self.llm = self._initialize_llm(model_name, use_4bit)
        
        # Initialize summary model
        logger.info("Loading summary model")
        self.summary_model = pipeline(
            "summarization",
            model="philschmid/bart-large-cnn-samsum",
            device=0 if self.device == "cuda" else -1
        )
        logger.info("Summary model loaded successfully")
        
        # Initialize FlowManager
        logger.info("Initializing FlowManager")
        self.flow_manager = FlowManager(self.llm)
        
        # Setup conversation memory with LangChain
        self.memory = ConversationBufferMemory(
            return_messages=True,
            input_key="input"
        )
        
        # Create conversation prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["history", "input", "past_context", "emotion_context", "guidelines"],
            template="""You are a supportive and empathetic mental health conversational AI. Your role is to provide therapeutic support while maintaining professional boundaries.

Previous conversation:
{history}

EMOTIONAL CONTEXT:
{emotion_context}

Past context: {past_context}

Relevant therapeutic guidelines:
{guidelines}

Current message: {input}

Provide a supportive response that:
1. Validates the user's feelings without using casual greetings
2. Asks relevant follow-up questions
3. Maintains a conversational tone , professional and empathetic tone
4. Focuses on understanding and support
5. Avoids repeating previous responses

Response:"""
        )
        
        # Create the conversation chain
        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            memory=self.memory,
            verbose=False
        )
        
        # Setup embeddings for vector search
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device}
        )
        
        # Setup vector database for retrieving relevant past conversations
        if therapy_guidelines_path and os.path.exists(therapy_guidelines_path):
            self.setup_vector_db(therapy_guidelines_path)
        else:
            self.setup_vector_db(None)
        
        # Initialize conversation storage
        self.conversations = {}
        
        # Load existing session summaries
        self.session_summaries = {}
        self._load_existing_summaries()
        
        logger.info("All models and components initialized successfully")
        
    def _load_emotion_model(self):
        try:
            return pipeline(
                "text-classification",
                model="SamLowe/roberta-base-go_emotions",
                top_k=None,
                device_map="auto" if self.device == "cuda" else None
            )
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            # Fallback 
            return pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base", 
                return_all_scores=True,
                device_map="auto" if self.device == "cuda" else None
            )
    
    def _initialize_llm(self, model_name: str, use_4bit: bool):
        """Initialize the language model with proper configuration."""
        try:
            # Configure quantization if needed
            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None

            # Load base model
            logger.info(f"Loading base model: {model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            # Load tokenizer
            logger.info("Loading tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token

            # Load PEFT model from Hugging Face
            logger.info(f"Loading PEFT model from {self.peft_model_path}")
            model = PeftModel.from_pretrained(base_model, self.peft_model_path)
            logger.info("Successfully loaded PEFT model")

            # Create text generation pipeline
            text_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Create LangChain wrapper
            llm = HuggingFacePipeline(pipeline=text_generator)
            
            return model, tokenizer, llm
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def setup_vector_db(self, guidelines_path: str = None):
        
        logger.info("Setting up FAISS vector database")
        
        # Check if vector DB exists
        vector_db_exists = os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss"))
        
        if not vector_db_exists:
            # Load therapy guidelines 
            if guidelines_path and os.path.exists(guidelines_path):
                loader = TextLoader(guidelines_path)
                documents = loader.load()
                
                # Split documents into chunks with better overlap for context
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # Smaller chunks for more precise retrieval
                    chunk_overlap=100,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create and save the vector store
                self.vector_db = FAISS.from_documents(chunks, self.embeddings)
                self.vector_db.save_local(VECTOR_DB_PATH)
                logger.info("Successfully loaded and indexed therapy guidelines")
            else:
                # Initialize with empty vector DB
                self.vector_db = FAISS.from_texts(["Initial empty vector store"], self.embeddings)
                self.vector_db.save_local(VECTOR_DB_PATH)
                logger.warning("No guidelines file provided, using empty vector store")
        else:
            # Load existing vector DB
            self.vector_db = FAISS.load_local(VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded existing vector database")
    
    def _load_existing_summaries(self):
        """Load existing session summaries from disk"""
        if not os.path.exists(SUMMARIES_DIR):
            return
            
        for filename in os.listdir(SUMMARIES_DIR):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(SUMMARIES_DIR, filename), 'r') as f:
                        summary_data = json.load(f)
                        session_id = summary_data.get('session_id')
                        if session_id:
                            self.session_summaries[session_id] = summary_data
                except Exception as e:
                    logger.warning(f"Failed to load summary from {filename}: {e}")

    def detect_emotion(self, text: str) -> Dict[str, float]:
        """Detect emotions in the text using the emotion classifier."""
        try:
            results = self.emotion_classifier(text)[0]
            return {result['label']: result['score'] for result in results}
        except Exception as e:
            logger.error(f"Error detecting emotions: {e}")
            return {"neutral": 1.0}

    def retrieve_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant past conversations using vector similarity"""
        if not hasattr(self, 'vector_db'):
            return ""
        
        try:
            # Retrieve similar documents from vector DB
            docs = self.vector_db.similarity_search(query, k=k)
            
            # Combine the content of retrieved documents
            relevant_context = "\n".join([doc.page_content for doc in docs])
            return relevant_context
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

    def retrieve_relevant_guidelines(self, query: str, emotion_context: str) -> str:
        """Retrieve relevant therapeutic guidelines """
        if not hasattr(self, 'vector_db'):
            return ""
        
        try:
            # Combine query and emotion context for better relevance
            search_query = f"{query} {emotion_context}"
            
            # Retrieve similar documents from vector DB
            docs = self.vector_db.similarity_search(search_query, k=2)
            
            # Combine the content of retrieved documents
            relevant_guidelines = "\n".join([doc.page_content for doc in docs])
            return relevant_guidelines
        except Exception as e:
            logger.error(f"Error retrieving guidelines: {e}")
            return ""

    def generate_response(self, prompt: str, emotion_data: Dict[str, float], conversation_history: List[Dict]) -> str:
        
        # Get primary and secondary emotions
        sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
        primary_emotion = sorted_emotions[0][0] if sorted_emotions else "neutral"
        
        # Get secondary emotions (if any)
        secondary_emotions = []
        for emotion, score in sorted_emotions[1:3]:  # Get 2nd and 3rd strongest emotions
            if score > 0.2:  # Only include if reasonably strong
                secondary_emotions.append(emotion)
        
        # Create emotion context string
        emotion_context = f"User is primarily feeling {primary_emotion}"
        if secondary_emotions:
            emotion_context += f" with elements of {' and '.join(secondary_emotions)}"
        emotion_context += "."
        
        # Retrieve relevant guidelines
        guidelines = self.retrieve_relevant_guidelines(prompt, emotion_context)
        
        # Retrieve past context
        past_context = self.retrieve_relevant_context(prompt)
        
        # Generate response using the conversation chain
        response = self.conversation.predict(
            input=prompt,
            past_context=past_context,
            emotion_context=emotion_context,
            guidelines=guidelines
        )
        
        # Clean up the response to only include the actual message
        response = response.split("Response:")[-1].strip()
        response = response.split("---")[0].strip()
        response = response.split("Note:")[0].strip()
        
        # Remove any casual greetings like "Hey" or "Hi"
        response = re.sub(r'^(Hey|Hi|Hello|Hi there|Hey there),\s*', '', response)
        
        # Ensure the response is unique and not repeating previous messages
        if len(conversation_history) > 0:
            last_responses = [msg["text"] for msg in conversation_history[-4:] if msg["role"] == "assistant"]
            if response in last_responses:
                # Generate a new response with a different angle
                response = self.conversation.predict(
                    input=f"{prompt} (Please provide a different perspective)",
                    past_context=past_context,
                    emotion_context=emotion_context,
                    guidelines=guidelines
                )
                response = response.split("Response:")[-1].strip()
                response = re.sub(r'^(Hey|Hi|Hello|Hi there|Hey there),\s*', '', response)
        
        return response.strip()

    def generate_session_summary(
        self,
        flow_manager_session: Dict = None
    ) -> Dict:
        
        if not flow_manager_session:
            return {
                "session_id": "",
                "user_id": "",
                "start_time": "",
                "end_time": datetime.now().isoformat(),
                "duration_minutes": 0,
                "current_phase": "unknown",
                "primary_emotions": [],
                "emotion_progression": [],
                "summary": "Error: No session data provided",
                "recommendations": ["Unable to generate recommendations"],
                "session_characteristics": {}
            }
            
        # Get session data from FlowManager
        session_id = flow_manager_session.get('session_id', '')
        user_id = flow_manager_session.get('user_id', '')
        current_phase = flow_manager_session.get('current_phase')
        
        if current_phase:
            # Convert ConversationPhase to dict
            current_phase = {
                'name': current_phase.name,
                'description': current_phase.description,
                'goals': current_phase.goals,
                'started_at': current_phase.started_at,
                'ended_at': current_phase.ended_at,
                'completion_metrics': current_phase.completion_metrics
            }
        
        session_start = flow_manager_session.get('started_at')
        if isinstance(session_start, str):
            session_start = datetime.fromisoformat(session_start)
        session_duration = (datetime.now() - session_start).total_seconds() / 60 if session_start else 0
        
        # Get emotion progression and primary emotions
        emotion_progression = flow_manager_session.get('emotion_progression', [])
        emotion_history = flow_manager_session.get('emotion_history', [])
        
        # Extract primary emotions from emotion history
        primary_emotions = []
        if emotion_history:
            # Get the most frequent emotions
            emotion_counts = {}
            for entry in emotion_history:
                emotions = entry.get('emotions', {})
                if isinstance(emotions, dict):
                    primary = max(emotions.items(), key=lambda x: x[1])[0]
                    emotion_counts[primary] = emotion_counts.get(primary, 0) + 1
            
            # sort by frequency and get top 3
            primary_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            primary_emotions = [emotion for emotion, _ in primary_emotions]
        
        # get session 
        session_characteristics = flow_manager_session.get('llm_context', {}).get('session_characteristics', {})
        
        # prepare the text for summarization
        summary_text = f"""
        Session Overview:
        - Session ID: {session_id}
        - User ID: {user_id}
        - Phase: {current_phase.get('name', 'unknown') if current_phase else 'unknown'}
        - Duration: {session_duration:.1f} minutes
        
        Emotional Analysis:
        - Primary Emotions: {', '.join(primary_emotions) if primary_emotions else 'No primary emotions detected'}
        - Emotion Progression: {', '.join(emotion_progression) if emotion_progression else 'No significant emotion changes noted'}
        
        Session Characteristics:
        - Therapeutic Alliance: {session_characteristics.get('alliance_strength', 'N/A')}
        - Engagement Level: {session_characteristics.get('engagement_level', 'N/A')}
        - Emotional Pattern: {session_characteristics.get('emotional_pattern', 'N/A')}
        - Cognitive Pattern: {session_characteristics.get('cognitive_pattern', 'N/A')}
        
        Key Observations:
        - The session focused on {current_phase.get('description', 'general discussion') if current_phase else 'general discussion'}
        - Main emotional themes: {', '.join(primary_emotions) if primary_emotions else 'not identified'}
        - Session progress: {session_characteristics.get('progress_quality', 'N/A')}
        """
        
        # Generate summary using BART
        summary = self.summary_model(
            summary_text,
            max_length=150,
            min_length=50,
            do_sample=False
        )[0]['summary_text']
        
        # Generate recommendations using Llama
        recommendations_prompt = f"""
        Based on the following session summary, provide 2-3 specific recommendations for follow-up:
        
        {summary}
        
        Session Characteristics:
        - Therapeutic Alliance: {session_characteristics.get('alliance_strength', 'N/A')}
        - Engagement Level: {session_characteristics.get('engagement_level', 'N/A')}
        - Emotional Pattern: {session_characteristics.get('emotional_pattern', 'N/A')}
        - Cognitive Pattern: {session_characteristics.get('cognitive_pattern', 'N/A')}
        
        Recommendations should be:
        1. Actionable and specific
        2. Based on the session content
        3. Focused on next steps
        """
        
        recommendations = self.llm.invoke(recommendations_prompt)
        
       
        recommendations = recommendations.split('\n')
        recommendations = [r.strip() for r in recommendations if r.strip()]
        recommendations = [r for r in recommendations if not r.startswith(('Based on', 'Session', 'Recommendations'))]
        
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "start_time": session_start.isoformat() if isinstance(session_start, datetime) else str(session_start),
            "end_time": datetime.now().isoformat(),
            "duration_minutes": session_duration,
            "current_phase": current_phase.get('name', 'unknown') if current_phase else 'unknown',
            "primary_emotions": primary_emotions,
            "emotion_progression": emotion_progression,
            "summary": summary,
            "recommendations": recommendations,
            "session_characteristics": session_characteristics
        }

    def start_session(self, user_id: str) -> tuple[str, str]:
        """Start a new conversation session for a user."""
        # Generate session id
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize FlowManager session
        self.flow_manager.initialize_session(user_id)
        
        # Create a new conversation
        self.conversations[user_id] = Conversation(
            user_id=user_id,
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            is_active=True
        )
        
        # Clear conversation memory
        self.memory.clear()
        
        # Generate initial greeting and question
        initial_message = """Hello! I'm here to support you today. How have you been feeling since our last session? 
If this is our first time meeting, I'd love to know what brings you here today and how you're feeling right now."""
        
        # Add the initial message to conversation history
        assistant_message = Message(
            text=initial_message,
            timestamp=datetime.now().isoformat(),
            role="assistant"
        )
        self.conversations[user_id].messages.append(assistant_message)
        
        logger.info(f"Session started for user {user_id}")
        return session_id, initial_message

    def end_session(
        self, 
        user_id: str, 
        flow_manager: Optional[Any] = None
    ) -> Optional[Dict]:

        if user_id not in self.conversations or not self.conversations[user_id].is_active:
            return None
        
        conversation = self.conversations[user_id]
        conversation.is_active = False
        
        # Get FlowManager session data
        flow_manager_session = self.flow_manager.user_sessions.get(user_id)
        
        # Generate session summary
        try:
            session_summary = self.generate_session_summary(flow_manager_session)
            
            # Save summary to disk
            summary_path = os.path.join(SUMMARIES_DIR, f"{session_summary['session_id']}.json")
            with open(summary_path, 'w') as f:
                json.dump(session_summary, f, indent=2)
            
            # Store in memory
            self.session_summaries[session_summary['session_id']] = session_summary
            
            # Clear conversation memory
            self.memory.clear()
            
            return session_summary
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")
            return None

    def process_message(self, user_id: str, message: str) -> str:
        """Process a user message and generate a response."""
        # Check for risk flags 
        risk_keywords = ["suicide", "kill myself", "end my life", "self-harm", "hurt myself"]
        risk_detected = any(keyword in message.lower() for keyword in risk_keywords)
        
        # Create or get conversation
        if user_id not in self.conversations or not self.conversations[user_id].is_active:
            self.start_session(user_id)
        
        conversation = self.conversations[user_id]
        
        #  user message -> conversation history
        new_message = Message(
            text=message,
            timestamp=datetime.now().isoformat(),
            role="user"
        )
        conversation.messages.append(new_message)
        
        # For crisis 
        if risk_detected:
            logger.warning(f"Risk flag detected in session {user_id}")
            
            crisis_response = """ I'm really sorry you're feeling this way — it sounds incredibly heavy, and I want you to know that you're not alone.

You don't have to face this by yourself. Our app has licensed mental health professionals who are ready to support you. I can connect you right now if you'd like.

In the meantime, I'm here to listen and talk with you. You can also do grounding exercises or calming techniques with me if you prefer. Just say "help me calm down" or "I need a break."

Would you like to connect with a professional now, or would you prefer to keep talking with me for a bit? Either way, I'm here for you."""
            
            #  assistant response -> conversation history
            assistant_message = Message(
                text=crisis_response,
                timestamp=datetime.now().isoformat(),
                role="assistant"
            )
            conversation.messages.append(assistant_message)
            
            return crisis_response
        
        # Detect emotions
        emotions = self.detect_emotion(message)
        conversation.emotion_history.append(emotions)
        
        # Process message with FlowManager
        flow_context = self.flow_manager.process_message(user_id, message, emotions)
        
        # Format conversation history 
        conversation_history = []
        for msg in conversation.messages:
            conversation_history.append({
                "text": msg.text,
                "timestamp": msg.timestamp,
                "role": msg.role
            })
        
        # Generate response
        response_text = self.generate_response(message, emotions, conversation_history)
        
        # Generate a follow-up question if the response is too short
        if len(response_text.split()) < 20 and not response_text.endswith('?'):
            follow_up_prompt = f"""Based on the conversation so far:
{chr(10).join([f"{msg['role']}: {msg['text']}" for msg in conversation_history[-3:]])}

Generate a thoughtful follow-up question that:
1. Shows you're actively listening
2. Encourages deeper exploration
3. Maintains therapeutic rapport
4. Is open-ended and non-judgmental

Respond with just the question."""
            
            follow_up = self.llm.invoke(follow_up_prompt)
            response_text += f"\n\n{follow_up}"
        
        #  assistant response -> conversation history
        assistant_message = Message(
            text=response_text,
            timestamp=datetime.now().isoformat(),
            role="assistant"
        )
        conversation.messages.append(assistant_message)
        
        # Update context
        conversation.context.update({
            "last_emotion": emotions,
            "last_interaction": datetime.now().isoformat(),
            "flow_context": flow_context
        })
        
        # Store this interaction in vector database
        current_interaction = f"User: {message}\nChatbot: {response_text}"
        self.vector_db.add_texts([current_interaction])
        self.vector_db.save_local(VECTOR_DB_PATH)
        
        return response_text

    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a user."""
        if user_id not in self.conversations:
            return []
        
        conversation = self.conversations[user_id]
        history = []
        
        for i, msg in enumerate(conversation.messages):
            emotions = conversation.emotion_history[i//2] if i % 2 == 0 and i//2 < len(conversation.emotion_history) else {}
            history.append({
                "text": msg.text,
                "timestamp": msg.timestamp,
                "role": msg.role,
                "emotions": emotions if msg.role == "user" else {}
            })
            
        return history

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session summary by ID."""
        return self.session_summaries.get(session_id)
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all session summaries for a specific user."""
        user_sessions = []
        for session_id, summary in self.session_summaries.items():
            if summary.get('user_id') == user_id:
                user_sessions.append(summary)
        return user_sessions

if __name__ == "__main__":
    pass