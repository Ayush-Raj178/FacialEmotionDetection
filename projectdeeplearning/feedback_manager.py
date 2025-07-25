import cv2
import numpy as np
import time
from collections import deque, defaultdict
from datetime import datetime
import threading

# Try to import TTS, but make it optional
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("Warning: pyttsx3 not installed. Text-to-speech will be disabled.")
    TTS_AVAILABLE = False

class FeedbackManager:
    def __init__(self, alert_callback=None):
        # Real-time feedback
        self.emotion_history = defaultdict(lambda: deque(maxlen=30))  # Store last 30 seconds of emotions
        self.alert_callback = alert_callback
        self.alert_cooldown = 0
        self.alert_messages = {
            'sad': "You seem sad. Would you like to talk about something?",
            'angry': "I notice you seem upset. Take a deep breath.",
            'fear': "You look concerned. Is everything alright?",
            'surprise': "You seem surprised!"
        }
        
        # Initialize TTS if available
        self.tts_engine = None
        if TTS_AVAILABLE:
            self.init_tts()
        
        # Analytics
        self.session_start = time.time()
        self.emotion_timeline = []
        self.engagement_scores = []
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        self.frame_skip = 2  # Process every 2nd frame
        
        # Face tracking
        self.face_trackers = {}
        self.next_face_id = 1
    
    def init_tts(self):
        """Initialize text-to-speech engine"""
        if not TTS_AVAILABLE:
            return
            
        try:
            self.tts_engine = pyttsx3.init()
            # Set properties for better voice
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)  # First available voice
                self.tts_engine.setProperty('rate', 150)  # Speed of speech
            print("TTS engine initialized successfully")
        except Exception as e:
            print(f"Could not initialize TTS: {e}")
            self.tts_engine = None
    
    def speak(self, text):
        """Speak the given text (runs in a separate thread)"""
        if not TTS_AVAILABLE or not self.tts_engine:
            print(f"[TTS] {text}")
            return
            
        def _speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Error in TTS: {e}")
        
        # Start TTS in a separate thread to avoid blocking
        tts_thread = threading.Thread(target=_speak)
        tts_thread.daemon = True
        tts_thread.start()
    
    def analyze_frame(self, frame, faces, emotions):
        """Analyze frame and update feedback systems"""
        start_time = time.time()
        self.frame_count += 1
        
        # Skip frames for better performance
        if self.frame_count % self.frame_skip != 0:
            return
        
        # Update emotion history and check for alerts
        self._update_emotion_history(emotions)
        self._check_alerts()
        
        # Update analytics
        self._update_analytics(emotions)
        
        # Track performance
        processing_time = (time.time() - start_time) * 1000  # in ms
        self.processing_times.append(processing_time)
    
    def _update_emotion_history(self, emotions):
        """Update emotion history for analytics"""
        timestamp = time.time()
        for emotion, confidence in emotions:
            self.emotion_history[emotion].append((timestamp, confidence))
    
    def _check_alerts(self):
        """Check if any alerts should be triggered"""
        if self.alert_cooldown > 0:
            self.alert_cooldown -= 1
            return
            
        # Check for sustained negative emotions
        for emotion in ['sad', 'angry', 'fear']:
            if emotion in self.emotion_history:
                # Check if this emotion has been present for more than 5 seconds
                if len(self.emotion_history[emotion]) > 10:  # Assuming 2 FPS
                    self.trigger_alert(emotion)
                    self.alert_cooldown = 30  # 15 second cooldown (at 2 FPS)
                    break
    
    def trigger_alert(self, emotion):
        """Trigger an alert for the given emotion"""
        if emotion in self.alert_messages:
            message = self.alert_messages[emotion]
            print(f"ALERT: {message}")
            self.speak(message)
            
            if self.alert_callback:
                self.alert_callback(emotion, message)
    
    def _update_analytics(self, emotions):
        """Update analytics data"""
        timestamp = time.time() - self.session_start
        if emotions:
            # Record dominant emotion
            dominant_emotion = max(emotions, key=lambda x: x[1])[0]
            self.emotion_timeline.append((timestamp, dominant_emotion))
            
            # Calculate engagement score (simplified)
            positive_emotions = ['happy', 'surprise']
            negative_emotions = ['sad', 'angry', 'fear']
            
            positive_score = sum(score for emo, score in emotions if emo.lower() in positive_emotions)
            negative_score = sum(score for emo, score in emotions if emo.lower() in negative_emotions)
            
            engagement = (positive_score - negative_score + 1) / 2  # Normalize to 0-1
            self.engagement_scores.append((timestamp, engagement))
    
    def get_engagement_score(self):
        """Get current engagement score (0-1)"""
        if not self.engagement_scores:
            return 0.5  # Neutral
        return self.engagement_scores[-1][1]
    
    def get_emotion_summary(self, window_seconds=30):
        """Get summary of emotions in the last window_seconds"""
        now = time.time()
        recent_emotions = []
        
        for emotion, history in self.emotion_history.items():
            # Get emotions within the time window
            recent = [conf for ts, conf in history if now - ts <= window_seconds]
            if recent:
                recent_emotions.append((emotion, sum(recent) / len(recent)))
        
        return sorted(recent_emotions, key=lambda x: -x[1])
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        if not self.processing_times:
            return {
                'fps': 0,
                'avg_processing_time': 0,
                'frame_skip': self.frame_skip
            }
            
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        fps = 1000 / (avg_processing_time * self.frame_skip) if avg_processing_time > 0 else 0
        
        return {
            'fps': round(fps, 1),
            'avg_processing_time': round(avg_processing_time, 1),
            'frame_skip': self.frame_skip
        }
    
    def adjust_performance(self, target_fps=15):
        """Dynamically adjust performance settings"""
        metrics = self.get_performance_metrics()
        
        if metrics['fps'] < target_fps * 0.8:
            # Increase frame skip if we're below target FPS
            self.frame_skip = min(5, self.frame_skip + 1)
        elif metrics['fps'] > target_fps * 1.2 and self.frame_skip > 1:
            # Decrease frame skip if we have headroom
            self.frame_skip = max(1, self.frame_skip - 1)
    
    def get_analytics_summary(self):
        """Get a summary of the session analytics"""
        if not self.emotion_timeline:
            return {}
            
        # Calculate dominant emotion percentages
        emotion_counts = defaultdict(int)
        for _, emotion in self.emotion_timeline:
            emotion_counts[emotion] += 1
            
        total = len(self.emotion_timeline)
        emotion_percentages = {k: v/total for k, v in emotion_counts.items()}
        
        # Calculate average engagement
        avg_engagement = sum(score for _, score in self.engagement_scores) / len(self.engagement_scores) \
                        if self.engagement_scores else 0.5
        
        return {
            'session_duration': time.time() - self.session_start,
            'emotion_distribution': dict(sorted(emotion_percentages.items(), 
                                             key=lambda x: -x[1])),
            'average_engagement': avg_engagement,
            'frame_count': self.frame_count
        }
