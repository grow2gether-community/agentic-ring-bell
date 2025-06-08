import cv2
import numpy as np

# This function normalizes a face embedding vector.
def normalize(vec):
    return vec / np.linalg.norm(vec)

# This function queries the database to find the best match for a given face embedding.
def recognize_face(collection, embedding, threshold=0.45):
    if embedding is None:
        return "unknown", float('inf')
    embedding_normalized = normalize(embedding).tolist()
    if collection.count() == 0:
        return "unknown", float('inf')

    results = collection.query(query_embeddings=[embedding_normalized], n_results=1)
    if not results or not results["distances"] or not results["distances"][0]:
        return "unknown", float('inf')

    distance = results["distances"][0][0]
    return (results["metadatas"][0][0].get("name", "unknown"), distance) if distance <= threshold else (
    "unknown", distance)

# This function estimates the distance of a face from the camera.
def estimate_distance(bbox):
    KNOWN_FACE_HEIGHT_CM = 20.0
    FOCAL_LENGTH = 500
    MIN_FACE_HEIGHT_PX = 40
    height_px = abs(bbox[3] - bbox[1])
    return (KNOWN_FACE_HEIGHT_CM * FOCAL_LENGTH) / height_px if height_px >= MIN_FACE_HEIGHT_PX else float('inf')


# This is the main function for the detection cycle.
def detect_visitor(face_app, collection):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return None

    print("\n📷 Camera activated. Looking for visitor...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        for face in face_app.get(frame):
            bbox = face.bbox.astype(int)
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            if estimate_distance(bbox) < 100:
                print(f"✅ Visitor detected up close. Capturing...")
                name, dist = recognize_face(collection, face.embedding)
                result_text = f"Detected: {name} (Dist: {dist:.2f})"
                color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(display_frame, result_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                            2)
                cv2.imshow("Agentic Ringbell - Live View", display_frame)
                cv2.waitKey(2000)
                cap.release()
                cv2.destroyAllWindows()
                return name

        cv2.imshow("Agentic Ringbell - Live View", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None
