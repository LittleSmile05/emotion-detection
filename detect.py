import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import cv2

# Function to create model
def create_emotion_model():
    # Initialize CNN model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolutional Layer
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # 7 emotion classes
    
    return model

# Function to train the model
def train_emotion_model(train_dir, test_dir, epochs=30, batch_size=64):
    # Data generators
    train_data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    validation_data_gen = ImageDataGenerator(rescale=1./255)
    
    # Load training and validation data
    train_generator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    )
    
    validation_generator = validation_data_gen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    )
    
    # Create the model
    emotion_model = create_emotion_model()
    
    # Disable OpenCL to prevent runtime errors
    cv2.ocl.setUseOpenCL(False)
    
    # Learning rate schedule
    initial_learning_rate = 0.0001
    lr_schedule = ExponentialDecay(
        initial_learning_rate, 
        decay_steps=100000, 
        decay_rate=0.96
    )
    
    # Compile the model
    optimizer = Adam(learning_rate=lr_schedule)
    emotion_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    # Print model summary
    emotion_model.summary()
    
    # Train the model
    print("Training model...")
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    
    history = emotion_model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )
    
    # Save model architecture as JSON
    model_json = emotion_model.to_json()
    with open("emotion_model.json", "w") as json_file:
        json_file.write(model_json)
    
    # Save trained model weights
    emotion_model.save_weights('emotion_model.h5')
    print("Model saved to disk")
    
    return emotion_model, history

# Function to visualize training history
def visualize_training_history(history):
    # Extract accuracy and loss values
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Create subplots
    plt.figure(figsize=(12, 5))
    
    # Accuracy graph
    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss graph
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Function to load a pretrained model
def load_emotion_model():
    # Load model architecture from JSON file
    json_file = open('emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # Create model from JSON
    emotion_model = model_from_json(loaded_model_json)
    
    # Load weights into the model
    emotion_model.load_weights('emotion_model.h5')
    
    return emotion_model

# Function for real-time emotion detection
def detect_emotions_realtime():
    # Define emotion dictionary
    emotion_dict = {
        0: "Angry", 
        1: "Disgusted", 
        2: "Fearful", 
        3: "Happy", 
        4: "Neutral", 
        5: "Sad", 
        6: "Surprised"
    }
    
    # Load pretrained model
    emotion_model = load_emotion_model()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Resize frame for better processing
        frame = cv2.resize(frame, (1280, 720))
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face detector
        try:
            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            print("Error: Haar cascade file not found")
            break
        
        # Detect faces
        faces = face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.3, 
            minNeighbors=5
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            
            # Extract face region
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            
            try:
                # Resize face to 48x48 pixels
                roi_gray_frame = cv2.resize(roi_gray_frame, (48, 48))
                
                # Normalize and reshape for model input
                img = roi_gray_frame.astype('float32') / 255.0
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                img = np.expand_dims(img, axis=0)   # Add batch dimension
                
                # Predict emotion
                emotion_prediction = emotion_model.predict(img, verbose=0)
                maxindex = int(np.argmax(emotion_prediction))
                
                # Display emotion text
                cv2.putText(
                    frame,
                    emotion_dict[maxindex],
                    (x+5, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
            except Exception as e:
                print(f"Error processing face: {e}")
        
        # Display the resulting frame
        cv2.imshow('Emotion Detection', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    print("Emotion Detection System")
    print("-----------------------")
    print("1: Train model")
    print("2: Run real-time emotion detection")
    print("3: Exit")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        # Check if dataset directories exist
        train_dir = input("Enter path to training data directory: ")
        test_dir = input("Enter path to test data directory: ")
        
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            print("Error: Dataset directories not found")
            return
        
        # Train model
        epochs = int(input("Enter number of epochs (default: 30): ") or 30)
        emotion_model, history = train_emotion_model(train_dir, test_dir, epochs=epochs)
        
        # Visualize training history
        visualize_training_history(history)
    
    elif choice == '2':
        # Check if model files exist
        if not os.path.exists('emotion_model.json') or not os.path.exists('emotion_model.h5'):
            print("Error: Model files not found. Train the model first.")
            return
        
        # Run real-time emotion detection
        detect_emotions_realtime()
    
    elif choice == '3':
        print("Exiting...")
    
    else:
        print("Invalid choice")

# Execute main function if script is run directly
if __name__ == "__main__":
    main()
