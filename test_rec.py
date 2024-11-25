
# import os
# import numpy as np
# import librosa
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Input, Dropout, Lambda
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import backend as K
# from collections import Counter

# # Constants
# SAMPLE_RATE = 16000
# MFCC_FEATURES = 13
# DATASET_DIR = "audio_dataset"

# # Functions
# def extract_mfcc(file_path, sample_rate=SAMPLE_RATE, n_mfcc=MFCC_FEATURES):
#     """
#     Extract MFCC features from an audio file.
#     """
#     if not os.path.exists(file_path):
#         print(f"Error: File '{file_path}' does not exist.")
#         return None

#     try:
#         audio, sr = librosa.load(file_path, sr=sample_rate)
#         mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
#         return np.mean(mfcc.T, axis=0)
#     except Exception as e:
#         print(f"Error extracting MFCC from {file_path}: {e}")
#         return None


# def prepare_data(dataset_dir):
#     labels, features = [], []
#     for label in os.listdir(dataset_dir):
#         label_dir = os.path.join(dataset_dir, label)
#         if not os.path.isdir(label_dir):
#             continue
#         for file in os.listdir(label_dir):
#             file_path = os.path.join(label_dir, file)
#             if file.endswith(".wav"):
#                 mfcc = extract_mfcc(file_path)
#                 if mfcc is not None:
#                     features.append(mfcc)
#                     labels.append(label)
#     return np.array(features), np.array(labels)


# def augment_audio(file_path):
#     """
#     Augment audio by adding noise and changing speed.
#     """
#     try:
#         audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
#         # Add noise
#         noise = np.random.normal(0, 0.005, audio.shape)
#         noisy_audio = audio + noise
#         # Change speed
#         speed_audio = librosa.effects.time_stretch(audio, rate=0.9)  # Slow down
#         return [audio, noisy_audio, speed_audio]
#     except Exception as e:
#         print(f"Error augmenting audio from {file_path}: {e}")
#         return []


# def siamese_model(input_shape):
#     input_layer = Input(shape=input_shape)
#     shared_network = Sequential([
#         Dense(128, activation='relu'),
#         Dropout(0.2),
#         Dense(64, activation='relu'),
#         Dropout(0.2),
#         Dense(32, activation='relu')
#     ])
#     encoded = shared_network(input_layer)
#     return Model(input_layer, encoded)


# def generate_pairs(X, y):
#     pairs, labels = [], []
#     num_classes = len(np.unique(y))
#     digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
#     for idx in range(len(X)):
#         current_class = y[idx]
#         # Ensure current_class exists in digit_indices
#         if current_class >= len(digit_indices) or len(digit_indices[current_class]) == 0:
#             continue
#         # Positive pair
#         pos_idx = np.random.choice(digit_indices[current_class])
#         pairs.append([X[idx], X[pos_idx]])
#         labels.append(1)
#         # Negative pair
#         negative_classes = list(set(range(num_classes)) - {current_class})
#         if negative_classes:
#             neg_class = np.random.choice(negative_classes)
#             if len(digit_indices[neg_class]) > 0:
#                 neg_idx = np.random.choice(digit_indices[neg_class])
#                 pairs.append([X[idx], X[neg_idx]])
#                 labels.append(0)
#     return np.array(pairs), np.array(labels)


# def authenticate(audio_file, reference_file, model, threshold=5.0):
#     """
#     Authenticate a speaker using the trained Siamese network.
#     """
#     if not os.path.exists(audio_file):
#         print(f"Error: Test file '{audio_file}' does not exist.")
#         return
#     if not os.path.exists(reference_file):
#         print(f"Error: Reference file '{reference_file}' does not exist.")
#         return

#     audio_mfcc = extract_mfcc(audio_file)
#     ref_mfcc = extract_mfcc(reference_file)

#     if audio_mfcc is None or ref_mfcc is None:
#         print("Error: Failed to extract MFCC from one or more files.")
#         return

#     audio_mfcc = audio_mfcc.reshape(1, -1)
#     ref_mfcc = ref_mfcc.reshape(1, -1)

#     distance = model.predict([audio_mfcc, ref_mfcc])[0][0]
#     print(f"Distance: {distance}")
#     if distance < threshold:  # Adjust the threshold as needed
#         print("Match: The speaker is authenticated.")
#     else:
#         print("No Match: Speaker not recognized.")


# # Main Script
# if __name__ == "__main__":
#     if not os.path.exists(DATASET_DIR):
#         print(f"Dataset directory {DATASET_DIR} not found.")
#         exit()

#     print("Augmenting data...")
#     augmented_data, augmented_labels = [], []
#     for label in os.listdir(DATASET_DIR):
#         label_dir = os.path.join(DATASET_DIR, label)
#         if not os.path.isdir(label_dir):
#             continue
#         for file in os.listdir(label_dir):
#             file_path = os.path.join(label_dir, file)
#             if file.endswith(".wav"):
#                 augmented_audios = augment_audio(file_path)
#                 for audio in augmented_audios:
#                     mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)
#                     augmented_data.append(np.mean(mfcc.T, axis=0))
#                     augmented_labels.append(label)

#     print("Preparing data...")
#     X, y = np.array(augmented_data), np.array(augmented_labels)
#     print(f"Data prepared. Found {len(X)} samples across {len(np.unique(y))} classes.")
#     print(f"Class distribution: {Counter(y)}")

#     print("Encoding labels...")
#     encoder = LabelEncoder()
#     y = encoder.fit_transform(y)

#     print("Splitting dataset...")
#     #Convert the test size from 0.5 to 0.2  when u collect audio files for around 20
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.5, random_state=42, stratify=y
#     )

#     print(f"Class distribution in training set: {Counter(y_train)}")
#     print(f"Class distribution in test set: {Counter(y_test)}")

#     print("Generating training pairs...")
#     pairs_train, labels_train = generate_pairs(X_train, y_train)
#     pairs_test, labels_test = generate_pairs(X_test, y_test)

#     print("Building model...")
#     input_shape = (MFCC_FEATURES,)
#     left_input = Input(input_shape)
#     right_input = Input(input_shape)
#     base_network = siamese_model(input_shape)
#     encoded_l = base_network(left_input)
#     encoded_r = base_network(right_input)
#     distance = Lambda(lambda x: K.sqrt(K.maximum(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True), K.epsilon())))([encoded_l, encoded_r])
#     siamese_net = Model([left_input, right_input], distance)
#     siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

#     print("Training model...")
#     siamese_net.fit(
#         [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
#         batch_size=32, epochs=10, validation_data=([pairs_test[:, 0], pairs_test[:, 1]], labels_test)
#     )
#     siamese_net.save("siamese_voice_model.h5")

#     print("Authenticating...")
#     reference_file = "audio_dataset/speaker1/nan1.wav"
#     test_file = "audio_dataset/speaker2/nan2.wav"
#     authenticate(test_file, reference_file, siamese_net)


# pip install numpy librosa sounddevice tensorflow keras   
# pip install numpy librosa tensorflow keras matplotlib sounddevice soundfile scikit-learn mtcnn keras-facenet opencv-python  
import os
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Constants
SAMPLE_RATE = 16000
DURATION = 300 # Recording duration in seconds
MFCC_FEATURES = 13

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print("Recording audio... Speak now.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    print("Recording complete.")
    return audio.flatten()

def extract_mfcc_from_audio(audio, sample_rate=SAMPLE_RATE, n_mfcc=MFCC_FEATURES):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None

# Custom function for Euclidean distance
def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def authenticate_live(model, reference_mfcc, threshold=5.0):
    live_audio = record_audio()
    live_mfcc = extract_mfcc_from_audio(live_audio)
    if live_mfcc is None:
        print("Error: Could not extract MFCC from live audio.")
        return

    live_mfcc = live_mfcc.reshape(1, -1)
    reference_mfcc = reference_mfcc.reshape(1, -1)

    distance = model.predict([live_mfcc, reference_mfcc])[0][0]
    print(f"Distance: {distance}")
    if distance < threshold:
        print("Match: The speaker is authenticated.")
    else:
        print("No Match: Speaker not recognized.")

if __name__ == "__main__":
    # Load the trained model with the custom function
    model_path = "siamese_voice_model.keras"
    if not os.path.exists(model_path):
        print(f"Trained model not found at '{model_path}'. Train the model first.")
        exit()

    siamese_net = load_model(model_path, custom_objects={"euclidean_distance": euclidean_distance})

    # Load reference audio
    reference_file = "audio_dataset/speaker1/nan1.wav"
    if not os.path.exists(reference_file):
        print(f"Error: Reference file '{reference_file}' does not exist.")
        exit()

    reference_audio, _ = librosa.load(reference_file, sr=SAMPLE_RATE)
    reference_mfcc = extract_mfcc_from_audio(reference_audio)
    if reference_mfcc is None:
        print("Error: Failed to extract MFCC from the reference file.")
        exit()

    authenticate_live(siamese_net, reference_mfcc)
