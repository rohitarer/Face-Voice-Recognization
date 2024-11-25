# pip install mtcnn keras_facenet tensorflow opencv-python numpy
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os

# Initialize MTCNN for face detection and FaceNet for face recognition
detector = MTCNN()
embedder = FaceNet()

# Known face embedding and name
known_face_embedding = None
known_face_name = "Authorized Person"

def align_face(image, keypoints):
    """
    Align face using keypoints (eyes).
    """
    try:
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # Validate the keypoints
        if not isinstance(left_eye, (tuple, list)) or not isinstance(right_eye, (tuple, list)):
            raise ValueError(f"Invalid keypoints: {keypoints}")
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        center = tuple(np.mean([left_eye, right_eye], axis=0).astype("int"))
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_face = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
        return aligned_face
    except Exception as e:
        print(f"Error aligning face: {e}")
        return image  # Return the original image if alignment fails

def load_reference_images(image_paths, name="Authorized Person"):
    """
    Load multiple reference images, compute embeddings, and average them.
    """
    global known_face_embedding, known_face_name
    embeddings = []
    try:
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb_image)

            if not results:
                print(f"No face detected in {image_path}")
                continue

            x, y, width, height = results[0]['box']
            keypoints = results[0]['keypoints']
            
            # Align the face if possible
            if "left_eye" in keypoints and "right_eye" in keypoints:
                aligned_face = align_face(rgb_image, keypoints)
            else:
                print(f"Keypoints missing or invalid: {keypoints}")
                aligned_face = rgb_image
            
            face = aligned_face[y:y+height, x:x+width]
            face = cv2.resize(face, (160, 160))
            embedding = embedder.embeddings([face])[0]
            embeddings.append(embedding)

        if embeddings:
            known_face_embedding = np.mean(embeddings, axis=0)  # Average embedding
            known_face_name = name
            print("Loaded multiple reference images. Generated averaged embedding.")
        else:
            print("No valid embeddings found in reference images.")
    except Exception as e:
        print(f"Error loading reference images: {e}")

def recognize_face(image):
    """
    Recognize faces in the given frame.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_image)
    
    for result in results:
        x, y, width, height = result['box']
        keypoints = result['keypoints']
        
        # Align the face if possible
        if "left_eye" in keypoints and "right_eye" in keypoints:
            aligned_face = align_face(rgb_image, keypoints)
        else:
            print(f"Keypoints missing or invalid: {keypoints}")
            aligned_face = rgb_image
        
        face = aligned_face[y:y+height, x:x+width]
        face = cv2.resize(face, (160, 160))
        face_embedding = embedder.embeddings([face])[0]
        
        if known_face_embedding is None:
            print("No reference embedding available. Skipping recognition.")
            continue
        
        # Compute distance between embeddings
        distance = np.linalg.norm(known_face_embedding - face_embedding)
        print(f"Recognition distance: {distance}")  # Debugging output
        
        # Recognition threshold
        if distance < 0.7:  # Adjusted threshold
            label = known_face_name
            color = (0, 255, 0)  # Green for recognized
        else:
            label = "Unknown"
            color = (0, 0, 255)  # Red for unrecognized
        
        # Draw rectangle and label
        cv2.rectangle(image, (x, y), (x+width, y+height), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return image

if __name__ == "__main__":
    # Paths to multiple reference images
    reference_image_paths = [
        "person1.1.jpg", "person1.2.jpg", "person1.3.jpg", "person1.4.jpg", "person2.1.jpeg", "person2.2.jpeg", "person2.3.jpeg","person2.4.jpeg"
    ]
    load_reference_images(reference_image_paths)

    if known_face_embedding is None:
        print("Failed to load the reference images. Exiting.")
    else:
        print("Reference images loaded successfully.")
        
        # Open the webcam
        cap = cv2.VideoCapture(0)  # Try 0 or 1 for webcam index
        if not cap.isOpened():
            print("Error: Webcam not accessible. Exiting.")
        else:
            print("Press 'q' to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame. Exiting.")
                    break
                
                frame = recognize_face(frame)
                cv2.imshow("Face Recognition", frame)  # For local environments
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break

            cap.release()
            cv2.destroyAllWindows()
