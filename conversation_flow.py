import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

class TopicInfo(BaseModel):
    """Represents a conversation topic with its metadata."""
    name: str
    confidence: float
    first_mentioned: str  # ISO timestamp
    last_mentioned: str  # ISO timestamp
    mention_count: int = 1
    related_emotions: Dict[str, float] = {}
    
class ConversationPhase(BaseModel):
    """Model representing a conversation phase."""
    name: str
    description: str
    goals: List[str]
    typical_duration: int  # in minutes
    started_at: Optional[str] = None  # ISO timestamp
    ended_at: Optional[str] = None  # ISO timestamp
    completion_metrics: Dict[str, float] = {}  # e.g., {'goal_progress': 0.8}

class FlowManager:
    """
    Advanced conversation flow manager for therapeutic conversations.
    Uses LLM for dynamic phase transitions and response guidance.
    """
    
    # Define conversation phases 
    PHASES = {
        'introduction': {
            'description': 'Establishing rapport and identifying main concerns',
            'goals': [
                'build therapeutic alliance',
                'identify primary concerns',
                'understand client expectations',
                'establish session structure'
            ],
            'typical_duration': 5  # In mins
        },
        'exploration': {
            'description': 'In-depth exploration of issues and their context',
            'goals': [
                'examine emotional responses',
                'explore thought patterns',
                'identify behavioral patterns',
                'understand situational context',
                'recognize relationship dynamics'
            ],
            'typical_duration': 15  # In mins
        },
        'intervention': {
            'description': 'Providing strategies, insights, and therapeutic interventions',
            'goals': [
                'introduce coping techniques',
                'reframe negative thinking',
                'provide emotional validation',
                'offer perspective shifts',
                'suggest behavioral modifications'
            ],
            'typical_duration': 20  # In minutes
        },
        'conclusion': {
            'description': 'Summarizing insights and establishing next steps',
            'goals': [
                'review key insights',
                'consolidate learning',
                'identify action items',
                'set intentions',
                'provide closure'
            ],
            'typical_duration': 5  # In minutes
        }
    }
    
    def __init__(self, llm, session_duration: int = 45):
       
        self.llm = llm
        self.session_duration = session_duration * 60  # Convert to seconds
        
        # User session data structures
        self.user_sessions = {}  # user_id -> session data
        
        logger.info(f"Initialized FlowManager with {session_duration} minute sessions")
    
    def _ensure_user_session(self, user_id: str):
        
        if user_id not in self.user_sessions:
            self.initialize_session(user_id)
    
    def initialize_session(self, user_id: str):
        
        now = datetime.now().isoformat()
        
        # Create initial phase
        initial_phase = ConversationPhase(
            name='introduction',
            description=self.PHASES['introduction']['description'],
            goals=self.PHASES['introduction']['goals'],
            typical_duration=self.PHASES['introduction']['typical_duration'],
            started_at=now
        )
        
        # Generate session ID
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize session data
        self.user_sessions[user_id] = {
            'session_id': session_id,
            'user_id': user_id,
            'started_at': now,
            'updated_at': now,
            'current_phase': initial_phase,
            'phase_history': [initial_phase],
            'message_count': 0,
            'emotion_history': [],
            'emotion_progression': [],
            'flags': {
                'crisis_detected': False,
                'long_silences': False
            },
            'llm_context': {
                'session_characteristics': {}
            }
        }
        
        logger.info(f"Initialized new session for user {user_id}")
        return self.user_sessions[user_id]
    
    def process_message(self, user_id: str, message: str, emotions: Dict[str, float]) -> Dict[str, Any]:

        self._ensure_user_session(user_id)
        session = self.user_sessions[user_id]
        
        # Update session
        now = datetime.now().isoformat()
        session['updated_at'] = now
        session['message_count'] += 1
        
        # Track emotions
        emotion_entry = {
            'timestamp': now,
            'emotions': emotions,
            'message_idx': session['message_count']
        }
        session['emotion_history'].append(emotion_entry)
        
        # Update emotion progression
        if not session.get('emotion_progression'):
            session['emotion_progression'] = []
        
        # Get primary emotion (highest confidence)
        primary_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        session['emotion_progression'].append(primary_emotion)
        
        # Check for phase transition
        self._check_phase_transition(user_id, message, emotions)
        
        # Update session characteristics via LLM analysis (periodically)
        if session['message_count'] % 5 == 0:
            self._update_session_characteristics(user_id)
        
        # Create flow context for response generation
        flow_context = self._create_flow_context(user_id)
        
        return flow_context
    
    def _check_phase_transition(self, user_id: str, message: str, emotions: Dict[str, float]):
       
        session = self.user_sessions[user_id]
        current_phase = session['current_phase']
        
        # Calculate session progress
        started_at = datetime.fromisoformat(session['started_at'])
        now = datetime.now()
        elapsed_seconds = (now - started_at).total_seconds()
        session_progress = elapsed_seconds / self.session_duration
        
        # Create prompt for LLM to evaluate phase transition
        phase_context = {
            'current': current_phase.name,
            'description': current_phase.description,
            'goals': current_phase.goals,
            'time_in_phase': (now - datetime.fromisoformat(current_phase.started_at)).total_seconds() / 60,
            'session_progress': session_progress,
            'message_count': session['message_count']
        }
        
        # Only check for transition if we've spent some time in current phase
        min_time_in_phase_minutes = max(2, current_phase.typical_duration * 0.5)
        if phase_context['time_in_phase'] < min_time_in_phase_minutes:
            return
            
        prompt = f"""
        Evaluate whether this therapeutic conversation should transition to the next phase.
        
        Current conversation state:
        - Current phase: {current_phase.name} ("{current_phase.description}")
        - Goals for this phase: {', '.join(current_phase.goals)}
        - Time spent in this phase: {phase_context['time_in_phase']:.1f} minutes
        - Session progress: {session_progress * 100:.1f}% complete
        - Message count: {session['message_count']}
        
        Latest message from user: "{message}"
        
        Current emotions: {', '.join([f"{e} ({score:.2f})" for e, score in 
                          sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]])}
        
        Phases in a therapeutic conversation:
        1. introduction: {self.PHASES['introduction']['description']}
        2. exploration: {self.PHASES['exploration']['description']}
        3. intervention: {self.PHASES['intervention']['description']}
        4. conclusion: {self.PHASES['conclusion']['description']}
        
        Consider:
        1. Have the goals of the current phase been sufficiently addressed?
        2. Is the timing appropriate considering overall session progress?
        3. Is there a natural transition point in the conversation?
        4. Does the emotional content suggest readiness to move forward?
        
        First, provide your analysis of whether the key goals of the current phase have been met.
        Then decide if the conversation should transition to the next phase.
        
        Respond with a JSON object in this format:
        {{
          "goals_progress": {{
            "goal1": 0.5,
            "goal2": 0.7
          }},
          "should_transition": false,
          "next_phase": "exploration",
          "reasoning": "brief explanation"
        }}
        
        Output ONLY valid JSON without additional text.
        """
        
        response = self.llm.invoke(prompt)
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                evaluation = json.loads(json_match.group(0))
                
                # Update goal progress metrics
                if 'goals_progress' in evaluation:
                    for goal, score in evaluation['goals_progress'].items():
                        if goal in current_phase.goals:
                            current_phase.completion_metrics[goal] = score
                
                # Check if we should transition
                if evaluation.get('should_transition', False):
                    next_phase_name = evaluation.get('next_phase')
                    if next_phase_name in self.PHASES:
                        self._transition_to_phase(user_id, next_phase_name, evaluation.get('reasoning', ''))
            except json.JSONDecodeError:
                self._check_time_based_transition(user_id)
        else:
            self._check_time_based_transition(user_id)
    
    def _check_time_based_transition(self, user_id: str):
        
        session = self.user_sessions[user_id]
        current_phase = session['current_phase']
        
        # Get elapsed time
        started_at = datetime.fromisoformat(session['started_at'])
        now = datetime.now()
        elapsed_minutes = (now - started_at).total_seconds() / 60
        
        # Calculate phase thresholds
        intro_threshold = self.PHASES['introduction']['typical_duration']
        explore_threshold = intro_threshold + self.PHASES['exploration']['typical_duration']
        intervention_threshold = explore_threshold + self.PHASES['intervention']['typical_duration']
        
        # Transition based on time
        next_phase = None
        if current_phase.name == 'introduction' and elapsed_minutes >= intro_threshold:
            next_phase = 'exploration'
        elif current_phase.name == 'exploration' and elapsed_minutes >= explore_threshold:
            next_phase = 'intervention'
        elif current_phase.name == 'intervention' and elapsed_minutes >= intervention_threshold:
            next_phase = 'conclusion'
            
        if next_phase:
            self._transition_to_phase(user_id, next_phase, "Time-based transition")
    
    def _transition_to_phase(self, user_id: str, next_phase_name: str, reason: str):
       
        session = self.user_sessions[user_id]
        current_phase = session['current_phase']
        
        # End current phase
        now = datetime.now().isoformat()
        current_phase.ended_at = now
        
        # Create new phase
        new_phase = ConversationPhase(
            name=next_phase_name,
            description=self.PHASES[next_phase_name]['description'],
            goals=self.PHASES[next_phase_name]['goals'],
            typical_duration=self.PHASES[next_phase_name]['typical_duration'],
            started_at=now
        )
        
        # Update session
        session['current_phase'] = new_phase
        session['phase_history'].append(new_phase)
        
        logger.info(f"User {user_id} transitioned from {current_phase.name} to {next_phase_name}: {reason}")
    
    def _update_session_characteristics(self, user_id: str):
        session = self.user_sessions[user_id]
        
        # Only do this periodically to save LLM calls
        if session['message_count'] < 5:
            return
            
        # Create a summary of the conversation so far
        message_sample = []
        emotion_summary = {}
        
        # Get recent messages
        for i, emotion_data in enumerate(session['emotion_history'][-10:]):
            msg_idx = emotion_data['message_idx']
            if i % 2 == 0:  # Just include a subset of messages
                message_sample.append(f"Message {msg_idx}: User emotions: {', '.join([f'{e}({s:.2f})' for e, s in sorted(emotion_data['emotions'].items(), key=lambda x: x[1], reverse=True)[:2]])}")
            
            # Aggregate emotions
            for emotion, score in emotion_data['emotions'].items():
                if score > 0.3:
                    emotion_summary[emotion] = emotion_summary.get(emotion, 0) + score
                    
        # Normalize emotion summary
        if emotion_summary:
            total = sum(emotion_summary.values())
            emotion_summary = {e: s/total for e, s in emotion_summary.items()}
            
        # Create prompt for LLM
        prompt = f"""
        Analyze this therapy session and provide a JSON response with the following characteristics:
        
        Current session state:
        - Phase: {session['current_phase'].name} ({session['current_phase'].description})
        - Message count: {session['message_count']}
        - Emotion summary: {', '.join([f'{e}({s:.2f})' for e, s in sorted(emotion_summary.items(), key=lambda x: x[1], reverse=True)])}
        
        Recent messages:
        {chr(10).join(message_sample)}
        
        Required JSON format:
        {{
          "alliance_strength": 0.8,
          "engagement_level": 0.7,
          "emotional_pattern": "brief description of emotional pattern",
          "cognitive_pattern": "brief description of cognitive pattern",
          "coping_mechanisms": ["mechanism1", "mechanism2"],
          "progress_quality": 0.6,
          "recommended_focus": "brief therapeutic recommendation"
        }}
        
        Important:
        1. Respond with ONLY the JSON object
        2. Use numbers between 0.0 and 1.0 for alliance_strength, engagement_level, and progress_quality
        3. Keep descriptions brief and focused
        4. Include at least 2 coping mechanisms
        5. Provide a specific recommended focus
        
        JSON Response:
        """
        
        response = self.llm.invoke(prompt)
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                characteristics = json.loads(json_match.group(0))
                # Validate required fields
                required_fields = [
                    'alliance_strength', 'engagement_level', 'emotional_pattern',
                    'cognitive_pattern', 'coping_mechanisms', 'progress_quality',
                    'recommended_focus'
                ]
                if all(field in characteristics for field in required_fields):
                    session['llm_context']['session_characteristics'] = characteristics
                    logger.info(f"Updated session characteristics for user {user_id}")
                else:
                    logger.warning("Missing required fields in session characteristics")
            except json.JSONDecodeError:
                logger.warning("Failed to parse session characteristics from LLM")
        else:
            logger.warning("No JSON object found in LLM response")
    
    def _create_flow_context(self, user_id: str) -> Dict[str, Any]:
        """Create flow context dict for response generation."""
        session = self.user_sessions[user_id]
        current_phase = session['current_phase']
        
        # Calculate session times
        started_at = datetime.fromisoformat(session['started_at'])
        now = datetime.now()
        elapsed_seconds = (now - started_at).total_seconds()
        remaining_seconds = max(0, self.session_duration - elapsed_seconds)
        
        # Get primary emotions
        emotions_summary = {}
        for emotion_data in session['emotion_history'][-3:]:  # Last 3 messages
            for emotion, score in emotion_data['emotions'].items():
                emotions_summary[emotion] = emotions_summary.get(emotion, 0) + score
                
        if emotions_summary:
            primary_emotions = sorted(emotions_summary.items(), key=lambda x: x[1], reverse=True)[:3]
        else:
            primary_emotions = []
        
        # Create guidance based on phase
        phase_guidance = []
        
        # Add phase-specific guidance
        if current_phase.name == 'introduction':
            phase_guidance.append("Build rapport and identify main concerns")
            if session['message_count'] > 3:
                phase_guidance.append("Begin exploring emotional context")
                
        elif current_phase.name == 'exploration':
            phase_guidance.append("Deepen understanding of issues and contexts")
            phase_guidance.append("Connect emotional patterns to identify themes")
                
        elif current_phase.name == 'intervention':
            phase_guidance.append("Offer support strategies and therapeutic insights")
            if remaining_seconds < 600:  # Less than 10 minutes left
                phase_guidance.append("Begin consolidating key insights")
                
        elif current_phase.name == 'conclusion':
            phase_guidance.append("Summarize insights and establish next steps")
            phase_guidance.append("Provide closure while maintaining supportive presence")
        
        # Add guidance based on session characteristics
        if 'session_characteristics' in session['llm_context']:
            char = session['llm_context']['session_characteristics']
            
            # Low alliance strength
            if char.get('alliance_strength', 0.8) < 0.6:
                phase_guidance.append("Focus on strengthening therapeutic alliance")
                
            # Low engagement
            if char.get('engagement_level', 0.8) < 0.6:
                phase_guidance.append("Increase engagement with more personalized responses")
                
            # Add recommended focus if available
            if 'recommended_focus' in char:
                phase_guidance.append(char['recommended_focus'])
        
        # Create flow context
        flow_context = {
            'phase': {
                'name': current_phase.name,
                'description': current_phase.description,
                'goals': current_phase.goals
            },
            'session': {
                'elapsed_minutes': elapsed_seconds / 60,
                'remaining_minutes': remaining_seconds / 60,
                'progress_percentage': (elapsed_seconds / self.session_duration) * 100,
                'message_count': session['message_count']
            },
            'emotions': [{'name': e, 'intensity': s} for e, s in primary_emotions],
            'guidance': phase_guidance
        }
        
        return flow_context 