from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime
import time
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from deepface import DeepFace
from face_database import FaceDatabase
import os
from feedback_manager import FeedbackManager

# Initialize face detector and model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('model.h5')

# Initialize face database
db = FaceDatabase()

# Emotion labels and colors
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_colors = {
    'Angry': (0, 0, 255),     # Red
    'Disgust': (0, 128, 0),    # Green
    'Fear': (255, 0, 0),       # Blue
    'Happy': (0, 255, 255),    # Yellow
    'Neutral': (255, 255, 255),# White
    'Sad': (128, 0, 128),      # Purple
    'Surprise': (0, 165, 255)  # Orange
}

# Initialize statistics
detection_history = []
emotion_counter = defaultdict(int)
start_time = time.time()
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'emotion_session_{session_id}.avi', fourcc, 20.0, (1280, 720))  # Increased resolution

def draw_emotion_stats(frame, emotions, x, y, w, h, name="Unknown"):
    # Constants for layout
    PADDING = 5
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1
    LINE_SPACING = 25
    
    # Calculate text sizes
    (name_width, name_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    
    # Draw name with background
    name_bg_top = max(0, y - name_height - 10)
    name_bg_bottom = y
    name_bg_left = x
    name_bg_right = x + name_width + 10
    
    cv2.rectangle(frame, (name_bg_left, name_bg_top), 
                 (name_bg_right, name_bg_bottom), (0, 0, 0), -1)
    cv2.putText(frame, name, (x + 5, y - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Calculate position for emotion stats (below the face)
    stats_x = x
    stats_y = y + h + 5
    
    # Find maximum text width for the emotion stats
    max_text_width = 0
    for emotion, prob in emotions:
        text = f"{emotion}: {prob:.1%}"
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                           FONT_SCALE, FONT_THICKNESS)
        max_text_width = max(max_text_width, text_width)
    
    # Draw emotion stats background (one rectangle for all stats)
    stats_bg_top = stats_y - 5
    stats_bg_bottom = stats_y + len(emotions) * LINE_SPACING - 5
    stats_bg_left = stats_x
    stats_bg_right = stats_x + max_text_width + 2 * PADDING
    
    cv2.rectangle(frame, (stats_bg_left, stats_bg_top),
                 (stats_bg_right, stats_bg_bottom), (0, 0, 0), -1)
    
    # Draw emotion probabilities
    for i, (emotion, prob) in enumerate(emotions):
        color = emotion_colors.get(emotion, (255, 255, 255))
        text = f"{emotion}: {prob:.1%}"
        text_y = stats_y + i * LINE_SPACING
        cv2.putText(frame, text, (stats_x + PADDING, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS)

def draw_session_stats(frame, counter, elapsed_time):
    # Constants for layout
    PANEL_WIDTH = 250
    PANEL_HEIGHT = 200
    PADDING = 10
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1
    LINE_HEIGHT = 25
    
    # Create semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (PADDING, PADDING), 
                 (PADDING + PANEL_WIDTH, PADDING + PANEL_HEIGHT), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw panel title
    title = "Session Stats"
    cv2.putText(frame, title, (PADDING + 10, PADDING + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw session ID
    cv2.putText(frame, f"ID: {session_id[-6:]}", 
               (PADDING + 10, PADDING + 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw emotion distribution
    y_offset = PADDING + 80
    cv2.putText(frame, "Emotion Distribution:", 
               (PADDING + 10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    total_detections = sum(counter.values()) or 1
    for i, (emotion, count) in enumerate(sorted(counter.items(), key=lambda x: -x[1])):
        if i >= 5:  # Limit to top 5 emotions
            break
        percentage = (count / total_detections) * 100
        color = emotion_colors.get(emotion, (255, 255, 255))
        text = f"{emotion}: {count} ({percentage:.1f}%)"
        cv2.putText(frame, text, 
                   (PADDING + 10, y_offset + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw FPS at the bottom
    fps = len(detection_history) / (elapsed_time + 1e-5)
    cv2.putText(frame, f"FPS: {fps:.1f}", 
               (PADDING + 10, PADDING + PANEL_HEIGHT - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def on_alert(emotion, message):
    print(f"Alert triggered: {emotion} - {message}")

feedback_manager = FeedbackManager(alert_callback=on_alert)

# Start camera with higher resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
frame_count = 0

print("Starting emotion detection with face recognition. Press 'q' to quit...")
print(f"Registered faces: {db.known_face_names}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        frame_count += 1
        
        # Skip frames for better performance
        if feedback_manager.frame_count % feedback_manager.frame_skip != 0:
            feedback_manager.frame_count += 1
            continue
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # If no faces detected with Haar, try with DeepFace
        if len(faces) == 0:
            try:
                # Save temporary frame for DeepFace
                temp_path = 'temp_frame.jpg'
                cv2.imwrite(temp_path, frame)
                
                # Get face detections with DeepFace
                detections = DeepFace.extract_faces(
                    img_path=temp_path,
                    enforce_detection=False,
                    detector_backend='mtcnn'  # More accurate than opencv
                )
                
                # Convert DeepFace detections to Haar format
                faces = []
                for detection in detections:
                    if detection['confidence'] > 0.9:  # Only consider high confidence detections
                        x, y, w, h = detection['facial_area'].values()
                        faces.append((x, y, w, h))
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"Error in DeepFace detection: {e}")
        
        # If we have faces, process them
        if len(faces) > 0:
            # Get face names using our database
            face_names = db.recognize_faces(frame, faces)
            
            # Process each detected face
            face_emotions = []
            for (x, y, w, h), name in zip(faces, face_names):
                # Get the face ROI for emotion detection
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # Get predictions
                    predictions = model.predict(roi, verbose=0)[0]
                    max_idx = np.argmax(predictions)
                    label = emotion_labels[max_idx]
                    confidence = predictions[max_idx]
                    
                    # Update statistics with name and emotion
                    emotion_counter[label] += 1
                    detection_history.append({
                        'frame': frame_count,
                        'name': name,
                        'emotion': label,
                        'confidence': float(confidence),
                        'timestamp': time.time() - start_time
                    })
                    
                    # Get top 3 emotions
                    top_emotions = sorted(zip(emotion_labels, predictions), 
                                        key=lambda x: -x[1])[:3]
                    
                    # Draw face rectangle with emotion color
                    color = emotion_colors.get(label, (0, 255, 0))
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw emotion label with confidence and name
                    label_text = f"{name}: {label} ({confidence:.1%})"
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(display_frame, (x, y-40), (x + text_width + 10, y), (0, 0, 0), -1)
                    cv2.putText(display_frame, label_text, (x+5, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw emotion statistics
                    draw_emotion_stats(display_frame, top_emotions, x, y, w, h, name)
                    
                    # Add to face emotions for feedback manager
                    face_emotions.append((label, confidence))
                
                # Add visual effect based on emotion
                if label == 'Happy':
                    cv2.circle(display_frame, (x + w//2, y - 30), 10, (0, 255, 255), -1)  # Sun
                elif label == 'Angry':
                    cv2.line(display_frame, (x, y-20), (x+20, y-30), (0, 0, 255), 2)  # Angry eyebrows
                    cv2.line(display_frame, (x+w-20, y-30), (x+w, y-20), (0, 0, 255), 2)
        
        # Update feedback manager with current frame data
        feedback_manager.analyze_frame(frame, faces, face_emotions)
        
        # Draw session statistics
        elapsed_time = time.time() - start_time
        draw_session_stats(display_frame, emotion_counter, elapsed_time)
        
        # Display performance metrics
        metrics = feedback_manager.get_performance_metrics()
        cv2.putText(display_frame, f"FPS: {metrics['fps']:.1f} | Skip: {metrics['frame_skip']}", 
                   (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Adjust performance dynamically
        feedback_manager.adjust_performance(target_fps=15)
        
        # Display engagement score
        engagement = feedback_manager.get_engagement_score()
        cv2.putText(display_frame, f"Engagement: {engagement*100:.1f}%", 
                   (display_frame.shape[1] - 150, display_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Write frame to video
        out.write(cv2.resize(display_frame, (640, 480)))
        
        # Display the frame
        cv2.imshow('Multi-Face Emotion Detector', display_frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Save session data
    df = pd.DataFrame(detection_history)
    df.to_csv(f'emotion_session_{session_id}.csv', index=False)
    
    # Before exiting, print analytics summary
    print("\nSession Summary:")
    analytics = feedback_manager.get_analytics_summary()
    
    # Handle missing keys gracefully
    duration = analytics.get('session_duration', time.time() - start_time)
    frame_count = analytics.get('frame_count', frame_count)
    emotion_dist = analytics.get('emotion_distribution', {emotion: count/sum(emotion_counter.values()) for emotion, count in emotion_counter.items()})
    avg_engagement = analytics.get('average_engagement', 0.5)
    
    print(f"Duration: {duration:.1f} seconds")
    print(f"Frames processed: {frame_count}")
    print("Emotion distribution:")
    for emotion, percentage in emotion_dist.items():
        print(f"- {emotion}: {percentage:.1%}")
    print(f"Average engagement: {avg_engagement*100:.1f}%")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
