import cv2
import numpy as np
import time

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


#Function to manage camera and get one frame
def get_camera_feed_and_detect(face_app, collection, cap):
    """
    Attempts to read one frame from the camera, detect faces, and recognize.
    Returns (frame_bytes, detected_name, status_message).
    """
    if not cap.isOpened():
        return None, None, "❌ Error: Camera not opened. Ensure it's not in use and permissions are granted."

    ret, frame = cap.read()
    if not ret:
        return None, None, "❌ Error: Failed to grab frame from camera."

    display_frame = frame.copy()
    faces = face_app.get(frame)

    current_frame_status = "No visitor detected yet."
    detected_name = None

    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2) # Blue box

        if estimate_distance(bbox) < 100: # Close enough for detection
            name, dist = recognize_face(collection, face.embedding)

            if name != "unknown":
                detected_name = name
                current_frame_status = f"✅ Visitor identified as: {detected_name} (Dist: {dist:.2f})"
                color = (0, 255, 0) # Green
            else:
                current_frame_status = f"Visitor detected (unknown). (Dist: {dist:.2f})"
                color = (0, 0, 255) # Red

            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(display_frame, f"{name} (Dist: {dist:.2f})", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            break # Process only the first close face

    ret, buffer = cv2.imencode('.jpg', display_frame)
    if ret:
        return buffer.tobytes(), detected_name, current_frame_status
    else:
        return None, None, "❌ Error encoding frame to JPEG."