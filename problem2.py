import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Assuming a predefined dataset
people = ['Person_A', 'Person_B', 'Person_C']  # Names of people in the dataset
dataset_dir = './voice_samples/'  # Directory where voice samples are stored

# Function to convert audio to Mel-spectrogram
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)  # Load the audio file
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB
    return mel_spec_db

# Load the dataset
X = []
y = []
for i, person in enumerate(people):
    person_dir = os.path.join(dataset_dir, person)
    for file_name in os.listdir(person_dir):
        file_path = os.path.join(person_dir, file_name)
        features = extract_features(file_path)
        X.append(features)
        y.append(i)  # Label the samples with the person's index

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Reshape X to fit into CNN (samples, width, height, channels)
X = X[..., np.newaxis]  # Add channel dimension

# Normalize the data
X = X / np.max(X)

# One-hot encode the labels
y = to_categorical(y, num_classes=len(people))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten and fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(people), activation='softmax'))  # Output layer for classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')

# Prediction function for new voice samples
def predict_person(audio_path):
    mel_spec = extract_features(audio_path)
    mel_spec = mel_spec[np.newaxis, ..., np.newaxis]  # Reshape to match input shape
    mel_spec = mel_spec / np.max(mel_spec)  # Normalize
    prediction = model.predict(mel_spec)
    predicted_person = people[np.argmax(prediction)]
    return predicted_person

# Example prediction
new_audio_sample = './voice_samples/Person_A/sample_001.wav'
predicted_person = predict_person(new_audio_sample)
print(f'Predicted person: {predicted_person}')