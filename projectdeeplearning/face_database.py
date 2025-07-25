import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace
from datetime import datetime

class FaceDatabase:
    def __init__(self, db_path='face_database.pkl'):
        self.db_path = db_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.similarity_threshold = 0.5  # Adjust this threshold as needed (lower is more strict)
        self.load_database()
    
    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                print(f"Loaded {len(self.known_face_names)} faces from database")
            except Exception as e:
                print(f"Error loading face database: {e}")
        else:
            print("No existing face database found. Starting fresh.")
    
    def save_database(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
        print(f"Database saved with {len(self.known_face_names)} faces")
    
    def add_face(self, image_path, name):
        """Add a new face to the database"""
        try:
            # Read and preprocess the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image at {image_path}")
                return False
                
            # Convert to RGB (DeepFace expects RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get face embedding using DeepFace
            try:
                # First try with the default detector (mtcnn is more accurate but slower)
                embedding_objs = DeepFace.represent(
                    img_path=img_rgb,
                    model_name='Facenet',
                    enforce_detection=True,
                    detector_backend='mtcnn'
                )
            except:
                # If mtcnn fails, try with opencv
                try:
                    embedding_objs = DeepFace.represent(
                        img_path=img_rgb,
                        model_name='Facenet',
                        enforce_detection=True,
                        detector_backend='opencv'
                    )
                except Exception as e:
                    print(f"Could not detect any face in {image_path}: {e}")
                    return False
            
            if not embedding_objs:
                print(f"No faces found in {image_path}")
                return False
                
            # Add the first face found
            self.known_face_encodings.append(embedding_objs[0]['embedding'])
            self.known_face_names.append(name)
            self.save_database()
            print(f"Successfully added {name} to the database")
            return True
            
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
    
    def recognize_faces(self, frame, face_locations):
        """Recognize faces in the given frame"""
        if not self.known_face_encodings:
            return ["Unknown"] * len(face_locations)
        
        # Convert face locations to DeepFace format
        faces = []
        for (x, y, w, h) in face_locations:
            # Expand the face area a bit for better recognition
            x = max(0, x - int(w * 0.1))
            y = max(0, y - int(h * 0.1))
            w = min(frame.shape[1] - x, int(w * 1.2))
            h = min(frame.shape[0] - y, int(h * 1.2))
            
            faces.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
        
        if not faces:
            return []
        
        # Get embeddings for all faces in the frame
        try:
            # Convert to RGB (DeepFace expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get embeddings for all faces at once
            embeddings = []
            for face in faces:
                # Extract face ROI
                x, y, w, h = face['x'], face['y'], face['w'], face['h']
                face_img = rgb_frame[y:y+h, x:x+w]
                
                # Get embedding for this face
                try:
                    embedding_obj = DeepFace.represent(
                        img_path=face_img,
                        model_name='Facenet',
                        enforce_detection=False,
                        detector_backend='skip'  # Skip detection since we already have face locations
                    )
                    if embedding_obj:
                        embeddings.append(embedding_obj[0]['embedding'])
                    else:
                        embeddings.append(None)
                except Exception as e:
                    print(f"Error getting embedding: {e}")
                    embeddings.append(None)
            
            # Match each face with known faces
            face_names = []
            for embedding in embeddings:
                if embedding is None:
                    face_names.append("Unknown")
                    continue
                
                # Calculate distances to all known faces
                distances = []
                for known_encoding in self.known_face_encodings:
                    # Calculate cosine similarity
                    similarity = np.dot(embedding, known_encoding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(known_encoding) + 1e-6
                    )
                    distances.append(1 - similarity)  # Convert to distance
                
                # Find the best match
                if distances and min(distances) < self.similarity_threshold:
                    best_match_idx = np.argmin(distances)
                    face_names.append(self.known_face_names[best_match_idx])
                else:
                    face_names.append("Unknown")
            
            return face_names
            
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return ["Unknown"] * len(face_locations)

def initialize_database():
    """Initialize the face database"""
    db = FaceDatabase()
    
    # Create a directory for known faces if it doesn't exist
    os.makedirs('known_faces', exist_ok=True)
    
    return db
