import cv2
import os
import shutil
from datetime import datetime
from face_database import FaceDatabase

def capture_face(name):
    """Capture and save a face image for the given name"""
    # Create directories if they don't exist
    os.makedirs('known_faces', exist_ok=True)
    
    # Initialize camera with a larger resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"\nPress 's' to save the image for {name}, 'q' to quit")
    print("Make sure the face is well-lit and clearly visible")
    
    # Create a window and move it to the front
    cv2.namedWindow('Face Registration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Registration', 800, 600)
    
    # Try to bring window to front (works on some systems)
    cv2.setWindowProperty('Face Registration', cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty('Face Registration', cv2.WND_PROP_TOPMOST, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Display the frame with instructions
        display_frame = frame.copy()
        
        # Add semi-transparent overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
        
        # Add text instructions
        cv2.putText(display_frame, f"Registering: {name}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 's' to save, 'q' to cancel", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Face Registration', display_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save the image
            # Create a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'known_faces/{name.lower()}_{timestamp}.jpg'
            
            # Save the image
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            
            # Show success message
            success_frame = frame.copy()
            cv2.putText(success_frame, "Image Saved!", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Face Registration', success_frame)
            cv2.waitKey(1000)  # Show success message for 1 second
            
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            return filename
            
        elif key == ord('q'):  # Quit without saving
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return None

def add_existing_faces(db):
    """Add all images from known_faces directory to the database"""
    if not os.path.exists('known_faces'):
        print("No 'known_faces' directory found.")
        return
        
    added = 0
    for filename in os.listdir('known_faces'):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Extract name from filename (assumes format: name_timestamp.jpg)
            name = '_'.join(filename.split('_')[:-1])
            if not name:
                name = filename.split('.')[0]  # Fallback to filename without extension
                
            image_path = os.path.join('known_faces', filename)
            if db.add_face(image_path, name):
                print(f"Added {name} from {filename}")
                added += 1
            else:
                print(f"Failed to add {filename}")
    
    if added > 0:
        db.save_database()
        print(f"\nSuccessfully added {added} faces to the database.")
    else:
        print("No new faces were added.")

def main():
    print("=== Face Registration Tool ===")
    
    # Initialize the face database
    db = FaceDatabase()
    
    while True:
        print("\nOptions:")
        print("1. Register a new face")
        print("2. Add all faces from 'known_faces' directory")
        print("3. List registered faces")
        print("4. Clear all registered faces")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            name = input("\nEnter person's name: ").strip()
            if not name:
                print("Name cannot be empty!")
                continue
                
            print(f"\nPosition your face in the camera and press 's' to capture...")
            print("Make sure your face is well-lit and clearly visible.")
            print("The camera window should appear now...")
            
            image_path = capture_face(name)
            
            if image_path:
                if db.add_face(image_path, name):
                    print(f"\nSuccessfully added {name} to the database!")
                else:
                    print("\nFailed to add face to the database. No face detected in the image.")
            
        elif choice == '2':
            print("\nAdding all faces from 'known_faces' directory...")
            add_existing_faces(db)
            
        elif choice == '3':
            print("\nRegistered faces:")
            if not db.known_face_names:
                print("No faces registered yet.")
            else:
                for i, name in enumerate(db.known_face_names, 1):
                    print(f"{i}. {name}")
        
        elif choice == '4':
            confirm = input("\nAre you sure you want to clear all registered faces? (y/n): ").lower()
            if confirm == 'y':
                db.known_face_encodings = []
                db.known_face_names = []
                db.save_database()
                print("All registered faces have been cleared.")
            
        elif choice == '5':
            print("\nGoodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
