import os
import uuid
import cv2
import numpy as np
import shutil
import tkinter as tk
from tkinter import filedialog

# Import the recognition function from the other module to check the main DB
from face_detection import recognize_face


# This function normalizes a face embedding vector.
def normalize_embedding(vec):
    if vec is None:
        return None
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


# This function adds a labeled face embedding to the ChromaDB collection.
def add_labeled_face_embedding(collection, name, embedding_list, crop_path):
    doc_id = str(uuid.uuid4())
    collection.add(
        ids=[doc_id],
        embeddings=[embedding_list],
        metadatas=[{"name": name}],
        documents=[crop_path]
    )
    print(f"Stored labeled face for '{name}' in ChromaDB.")


# This function gets image paths from the user using a file dialog.
def get_image_paths_from_user():
    all_image_paths = set()
    print("\n--- Image Enrollment ---")
    while True:
        input("Press Enter to open the file explorer and select images...")
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_paths = filedialog.askopenfilenames(
            parent=root,
            title="Select images for face enrollment",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        root.destroy()
        if file_paths:
            newly_added_count = len(set(file_paths) - all_image_paths)
            all_image_paths.update(file_paths)
            print(f"  > Added {newly_added_count} new image(s). Total unique images: {len(all_image_paths)}.")
        else:
            print("  > No new files were selected in this round.")
        while True:
            confirmation = input("Have you uploaded all the images? (yes/no): ").strip().lower()
            if confirmation in ["yes", "y"]:
                if not all_image_paths:
                    print("No images were provided.")
                return list(all_image_paths)
            elif confirmation in ["no", "n"]:
                break
            else:
                print("  Invalid input. Please enter 'yes' or 'no'.")


# === THIS IS THE MAIN CORRECTED FUNCTION ===
def run_enrollment_workflow(face_app, collection):
    """
    The main workflow for enrolling faces, now with DB suggestion logic.
    """
    image_paths_to_process = get_image_paths_from_user()
    if not image_paths_to_process:
        return

    session_unique_faces = []
    CROPPED_FACES_DIR = "detected_face_crops"
    if not os.path.exists(CROPPED_FACES_DIR):
        os.makedirs(CROPPED_FACES_DIR)

    # Constants for this enrollment session
    SESSION_GROUPING_DISTANCE_THRESHOLD = 0.35
    DB_SUGGESTION_RECOGNITION_THRESHOLD = 0.55

    for image_path in image_paths_to_process:
        img = cv2.imread(image_path)
        if img is None:
            continue

        for face_obj in face_app.get(img):
            emb_norm_np = normalize_embedding(face_obj.embedding)
            if emb_norm_np is None:
                continue

            # Check if this face is already known within this session
            is_new_in_session = all(
                1 - np.dot(emb_norm_np, np.array(uf['embedding'])) > SESSION_GROUPING_DISTANCE_THRESHOLD
                for uf in session_unique_faces
            )

            if is_new_in_session:
                # **FIX:** The function call now uses explicit keyword arguments for clarity and to prevent errors.
                name_from_db, dist_from_db = recognize_face(
                    collection=collection,
                    embedding=face_obj.embedding,
                    threshold=DB_SUGGESTION_RECOGNITION_THRESHOLD
                )

                # Create the crop and save it
                bbox = face_obj.bbox.astype(int)
                padding = 20
                x1, y1 = max(0, bbox[0] - padding), max(0, bbox[1] - padding)
                x2, y2 = min(img.shape[1], bbox[2] + padding), min(img.shape[0], bbox[3] + padding)
                crop_filename = os.path.join(CROPPED_FACES_DIR, f"session_unique_{uuid.uuid4().hex[:6]}.jpg")
                cv2.imwrite(crop_filename, img[y1:y2, x1:x2])

                # Add the new unique face along with its DB suggestion (if any)
                session_unique_faces.append({
                    'embedding': emb_norm_np.tolist(),
                    'crop_path': crop_filename,
                    'db_suggestion': name_from_db,
                    'db_suggestion_dist': dist_from_db
                })

    # --- Labeling Phase (Now with suggestion logic) ---
    if not session_unique_faces:
        print("\nNo new unique faces detected to enroll.")
    else:
        print(f"\n--- Time to Label {len(session_unique_faces)} Unique Face(s) ---")
        for face in session_unique_faces:
            print(f"\n  View face crop at: {face['crop_path']}")

            suggestion_text = ""
            if face['db_suggestion'] and face['db_suggestion'] != 'unknown':
                suggestion_text = f" (DB Suggests: '{face['db_suggestion']}' with dist: {face['db_suggestion_dist']:.3f})"

            while True:
                prompt_msg = f"  Who is this?{suggestion_text}\n  Enter name, or type 'skip', or 'use_db' to accept suggestion: "
                user_input = input(prompt_msg).strip()

                if user_input.lower() == 'skip':
                    print("  Skipped.")
                    break
                elif user_input.lower() == 'use_db' and face['db_suggestion'] and face['db_suggestion'] != 'unknown':
                    final_label = face['db_suggestion']
                    print(f"  Labeling as '{final_label}' (from DB suggestion).")
                    add_labeled_face_embedding(collection, final_label, face['embedding'], face['crop_path'])
                    break
                elif user_input:
                    final_label = user_input
                    print(f"  Labeling as '{final_label}'.")
                    add_labeled_face_embedding(collection, final_label, face['embedding'], face['crop_path'])
                    break
                else:
                    print("  Invalid input.")

    # Final cleanup
    if os.path.exists(CROPPED_FACES_DIR):
        shutil.rmtree(CROPPED_FACES_DIR)
    print("\nEnrollment complete.")
