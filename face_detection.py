import os
import uuid
import cv2
import numpy as np
import chromadb
from dotenv import load_dotenv
from chromadb.config import Settings
from insightface.app import FaceAnalysis

# === SETUP ===
load_dotenv()

# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# === Face Recognition Helper ===
def normalize(vec):
    return vec / np.linalg.norm(vec)

# === ChromaDB Setup for permenant storage ===
# client = chromadb.PersistentClient(
#     path="chroma_db_face_data", 
#     settings=Settings(anonymized_telemetry=False)
# )
# collection = client.get_collection(name="face_embeddings")

# === ChromaDB Setup for in-memory storage ===
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="face_embeddings", metadata={"hnsw:space": "cosine"})

# === Add Faces to DB ===
def add_known_face(name, image_path):
    img = cv2.imread(image_path)
    faces = face_app.get(img)
    if not faces:
        print(f"❌ No face found in {image_path}")
        return
    emb = normalize(faces[0].embedding).tolist()
    doc_id = str(uuid.uuid4())
    collection.add(
        ids=[doc_id],
        embeddings=[emb],
        metadatas=[{"name": name}],
        documents=[image_path]
    )
    print(f"✅ Added {name}'s face to ChromaDB.")

# Add Subbu's faces from s1.jpg to s12.jpg
for i in range(1, 9):
    add_known_face("subbu", f"./subbu{i}.jpg")

# === Recognize Face ===
def recognize_face(embedding, threshold=0.4):
    embedding = normalize(embedding).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=1)
    if results["distances"][0][0] > threshold:
        return None, results["distances"][0][0]
    matched_name = results["metadatas"][0][0]["name"]
    return matched_name, results["distances"][0][0]

# === Distance Estimation Constants ===
KNOWN_FACE_HEIGHT_CM = 20.0
FOCAL_LENGTH = 500
TRIGGER_DISTANCE_CM = 100
MIN_FACE_HEIGHT_PX = 40

def estimate_distance(bbox, known_height=KNOWN_FACE_HEIGHT_CM, focal_length=FOCAL_LENGTH):
    x1, y1, x2, y2 = bbox
    height_px = abs(y2 - y1)
    if height_px < MIN_FACE_HEIGHT_PX:
        return float('inf')
    return (known_height * focal_length) / height_px

#=== Main Detection Logic ===
def detect_face_and_greet():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("📷 Looking for visitors...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            distance_cm = estimate_distance(bbox)

            if distance_cm < TRIGGER_DISTANCE_CM:
                cap.release()
                cv2.destroyAllWindows()

                emb = face.embedding
                name, dist = recognize_face(emb)
                print(f"Detected: {name} with distance: {dist}")
                return name if name else "unknown"

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


print(detect_face_and_greet())