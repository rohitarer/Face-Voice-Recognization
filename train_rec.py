import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from collections import Counter

# Constants
SAMPLE_RATE = 16000
MFCC_FEATURES = 13
DATASET_DIR = "audio_dataset"

# Functions
def extract_mfcc(file_path, sample_rate=SAMPLE_RATE, n_mfcc=MFCC_FEATURES):
    """
    Extract MFCC features from an audio file.
    """
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error extracting MFCC from {file_path}: {e}")
        return None

def prepare_data(dataset_dir):
    """
    Prepare MFCC features and labels from the dataset.
    """
    labels, features = [], []
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            if file.endswith(".wav"):
                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(label)
    return np.array(features), np.array(labels)

def siamese_model(input_shape):
    """
    Build the Siamese Neural Network.
    """
    input_layer = Input(shape=input_shape)
    shared_network = Sequential([
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu')
    ])
    encoded = shared_network(input_layer)
    return Model(input_layer, encoded)

def generate_pairs(X, y):
    """
    Generate positive and negative pairs for training the Siamese network.
    """
    pairs, labels = [], []
    num_classes = len(np.unique(y))
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    for idx in range(len(X)):
        current_class = y[idx]
        if len(digit_indices[current_class]) == 0:
            continue
        # Positive pair
        pos_idx = np.random.choice(digit_indices[current_class])
        pairs.append([X[idx], X[pos_idx]])
        labels.append(1)
        # Negative pair
        negative_classes = list(set(range(num_classes)) - {current_class})
        if negative_classes:
            neg_class = np.random.choice(negative_classes)
            neg_idx = np.random.choice(digit_indices[neg_class])
            pairs.append([X[idx], X[neg_idx]])
            labels.append(0)
    return np.array(pairs), np.array(labels)

# Custom function for Euclidean distance
def euclidean_distance(vectors):
    """
    Custom function to compute Euclidean distance.
    """
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

# Main Script
if __name__ == "__main__":
    print("Preparing data...")
    X, y = prepare_data(DATASET_DIR)
    if len(X) == 0 or len(y) == 0:
        print("No valid audio files found in the dataset.")
        exit()

    print(f"Data prepared. Found {len(X)} samples across {len(np.unique(y))} classes.")
    print(f"Class distribution: {Counter(y)}")

    print("Encoding labels...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    print("Generating training pairs...")
    pairs_train, labels_train = generate_pairs(X_train, y_train)
    pairs_test, labels_test = generate_pairs(X_test, y_test)

    print("Building model...")
    input_shape = (MFCC_FEATURES,)
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    base_network = siamese_model(input_shape)

    # Use the custom Euclidean distance function
    distance = Lambda(
        euclidean_distance,
        output_shape=(1,)
    )([base_network(left_input), base_network(right_input)])

    siamese_net = Model([left_input, right_input], distance)
    siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

    print("Training model...")
    siamese_net.fit(
        [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
        batch_size=32, epochs=10, validation_data=([pairs_test[:, 0], pairs_test[:, 1]], labels_test)
    )

    # Save the model in native Keras format
    siamese_net.save("siamese_voice_model.keras")
    print("Model saved as 'siamese_voice_model.keras'.")
