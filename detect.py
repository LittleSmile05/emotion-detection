# First install all required packages
!pip install fer deepface opencv-python matplotlib tensorflow

import cv2
import matplotlib.pyplot as plt
from fer import FER
from deepface import DeepFace
import os

def detect_emotions_fer(image_path):
    """Detect emotions using FER library"""
    try:
        print("\nLoading FER model...")
        detector = FER(mtcnn=True)  # Using more accurate MTCNN detector
        
        print("Processing image with FER...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image at path: " + image_path)
            
        # Detect emotions
        result = detector.detect_emotions(image)
        
        if not result:
            print("No faces detected with FER")
            return
            
        # Process and display results
        print("\nFER Results:")
        for i, face in enumerate(result):
            emotions = face["emotions"]
            box = face["box"]
            dominant_emotion, dominant_score = max(emotions.items(), key=lambda x: x[1])
            
            print(f"\nFace {i+1} at position {box}:")
            for emotion, score in emotions.items():
                print(f"{emotion:8s}: {score:.2f}")
            print(f"Dominant emotion: {dominant_emotion} ({dominant_score:.2f})")
            
            # Draw rectangle and emotion text
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"{dominant_emotion}: {dominant_score:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("FER Emotion Detection")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"FER Error: {str(e)}")

def detect_emotions_deepface(image_path):
    """Detect emotions using DeepFace library"""
    try:
        print("\nLoading DeepFace model...")
        
        print("Processing image with DeepFace...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image at path: " + image_path)
            
        # Analyze emotions
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], 
                                enforce_detection=True, silent=True)
        
        # Process and display results
        print("\nDeepFace Results:")
        for i, face in enumerate(result):
            emotions = face['emotion']
            region = face['region']
            dominant_emotion, dominant_score = max(emotions.items(), key=lambda x: x[1])
            
            print(f"\nFace {i+1} at position {region}:")
            for emotion, score in emotions.items():
                print(f"{emotion:8s}: {score:.2f}")
            print(f"Dominant emotion: {dominant_emotion} ({dominant_score:.2f})")
            
            # Draw rectangle and emotion text
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, f"{dominant_emotion}: {dominant_score:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("DeepFace Emotion Detection")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"DeepFace Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Set the image path (for Google Colab)
    image_path = "/content/happy.jpg"
    
    # If running locally, you can use this instead:
    # image_path = input("Enter path to your image: ").strip()
    
    print(f"\nStarting emotion detection for image: {image_path}")
    print("Note: First run may take several minutes to download models.")
    
    # Verify the image exists
    if not os.path.exists(image_path):
        print(f"\nError: Image not found at {image_path}")
        print("Please ensure:")
        print("1. The file exists at that path")
        print("2. You've uploaded it to Colab if using Google Colab")
    else:
        # Run both detection methods
        detect_emotions_fer(image_path)
        detect_emotions_deepface(image_path)
    
    print("\nEmotion detection complete!")
