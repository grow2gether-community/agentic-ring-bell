# # import cv2
# # import numpy as np
# #
# # # This function normalizes a face embedding vector.
# # def normalize(vec):
# #     return vec / np.linalg.norm(vec)
# #
# # # This function queries the database to find the best match for a given face embedding.
# # def recognize_face(collection, embedding, threshold=0.45):
# #     if embedding is None:
# #         return "unknown", float('inf')
# #     embedding_normalized = normalize(embedding).tolist()
# #     if collection.count() == 0:
# #         return "unknown", float('inf')
# #
# #     results = collection.query(query_embeddings=[embedding_normalized], n_results=1)
# #     if not results or not results["distances"] or not results["distances"][0]:
# #         return "unknown", float('inf')
# #
# #     distance = results["distances"][0][0]
# #     return (results["metadatas"][0][0].get("name", "unknown"), distance) if distance <= threshold else (
# #     "unknown", distance)
# #
# # # This function estimates the distance of a face from the camera.
# # def estimate_distance(bbox):
# #     KNOWN_FACE_HEIGHT_CM = 20.0
# #     FOCAL_LENGTH = 500
# #     MIN_FACE_HEIGHT_PX = 40
# #     height_px = abs(bbox[3] - bbox[1])
# #     return (KNOWN_FACE_HEIGHT_CM * FOCAL_LENGTH) / height_px if height_px >= MIN_FACE_HEIGHT_PX else float('inf')
# #
# #
# # # This is the main function for the detection cycle.
# # def detect_visitor(face_app, collection):
# #     cap = cv2.VideoCapture(0)
# #     if not cap.isOpened():
# #         print("❌ Could not open webcam.")
# #         return None
# #
# #     print("\n📷 Camera activated. Looking for visitor...")
# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #
# #         display_frame = frame.copy()
# #         for face in face_app.get(frame):
# #             bbox = face.bbox.astype(int)
# #             cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
# #             if estimate_distance(bbox) < 100:
# #                 print(f"✅ Visitor detected up close. Capturing...")
# #                 name, dist = recognize_face(collection, face.embedding)
# #                 result_text = f"Detected: {name} (Dist: {dist:.2f})"
# #                 color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
# #                 cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
# #                 cv2.putText(display_frame, result_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
# #                             2)
# #                 cv2.imshow("Agentic Ringbell - Live View", display_frame)
# #                 cv2.waitKey(2000)
# #                 cap.release()
# #                 cv2.destroyAllWindows()
# #                 return name
# #
# #         cv2.imshow("Agentic Ringbell - Live View", display_frame)
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
# #
# #     cap.release()
# #     cv2.destroyAllWindows()
# #     return None
#
# import cv2
# import numpy as np
# import time # Import time for a small delay
#
# # This function normalizes a face embedding vector.
# def normalize(vec):
#     return vec / np.linalg.norm(vec)
#
# # This function queries the database to find the best match for a given face embedding.
# def recognize_face(collection, embedding, threshold=0.45):
#     if embedding is None:
#         return "unknown", float('inf')
#     embedding_normalized = normalize(embedding).tolist()
#     if collection.count() == 0:
#         return "unknown", float('inf')
#
#     results = collection.query(query_embeddings=[embedding_normalized], n_results=1)
#     if not results or not results["distances"] or not results["distances"][0]:
#         return "unknown", float('inf')
#
#     distance = results["distances"][0][0]
#     return (results["metadatas"][0][0].get("name", "unknown"), distance) if distance <= threshold else (
#     "unknown", distance)
#
# # This function estimates the distance of a face from the camera.
# def estimate_distance(bbox):
#     KNOWN_FACE_HEIGHT_CM = 20.0
#     FOCAL_LENGTH = 500
#     MIN_FACE_HEIGHT_PX = 40
#     height_px = abs(bbox[3] - bbox[1])
#     return (KNOWN_FACE_HEIGHT_CM * FOCAL_LENGTH) / height_px if height_px >= MIN_FACE_HEIGHT_PX else float('inf')
#
#
# # This is the main function for the detection cycle, modified to yield frames.
# def detect_visitor_stream(face_app, collection): # Renamed for clarity
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("❌ Could not open webcam.")
#         yield None, "Error: Could not open webcam." # Yield an error message
#         return
#
#     print("\n📷 Camera activated. Looking for visitor...")
#     yield None, "Camera activated. Looking for visitor..." # Initial status message
#
#     detected_name = None
#     # We will loop continuously until a face is detected and recognized
#     # Or until an external signal (from Streamlit) tells us to stop
#     while detected_name is None:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             yield None, "Error: Failed to grab frame."
#             break # Exit loop on frame error
#
#         display_frame = frame.copy()
#         faces = face_app.get(frame) # Get all faces in the frame
#
#         current_frame_status = "No visitor detected yet."
#         for face in faces:
#             bbox = face.bbox.astype(int)
#             cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2) # Blue box for any detected face
#
#             # Only process faces that are reasonably close
#             if estimate_distance(bbox) < 100: # Threshold for "close enough"
#                 name, dist = recognize_face(collection, face.embedding)
#
#                 # Determine if a recognized face qualifies as a "detected visitor"
#                 if name != "unknown":
#                     detected_name = name # Found a recognized visitor, exit the loop
#                     current_frame_status = f"✅ Visitor identified as: {detected_name} (Dist: {dist:.2f})"
#                     color = (0, 255, 0) # Green for recognized
#                 else:
#                     current_frame_status = f"Visitor detected (unknown). (Dist: {dist:.2f})"
#                     color = (0, 0, 255) # Red for unknown but detected
#
#                 # Draw final box and text on the frame
#                 cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
#                 cv2.putText(display_frame, f"{name} (Dist: {dist:.2f})", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#                 break # Process only the first close face for simplicity in this demo
#
#         # Yield the processed frame and current status
#         ret, buffer = cv2.imencode('.jpg', display_frame)
#         if ret:
#             yield buffer.tobytes(), current_frame_status
#         else:
#             yield None, "Error encoding frame."
#
#         time.sleep(0.01) # Small delay to prevent burning CPU
#
#     cap.release()
#     print("Camera released.")
#     cv2.destroyAllWindows()
#     return detected_name # Return the name if recognized, or None if the loop broke without recognition

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


# --- NEW/MODIFIED: Function to manage camera and get one frame ---
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