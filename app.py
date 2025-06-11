import streamlit as st
import cv2
import numpy as np
import time
import chromadb
from dotenv import load_dotenv
from chromadb.config import Settings
from insightface.app import FaceAnalysis
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os
import shutil
import uuid
import json

# Import functions/graph from your modules
# Ensure vbot_graph.py is updated to remove direct speak_text/listen_prompted calls from nodes
from vbot_graph import graph, speak_text, listen_for_audio_once, SessionState

# Import other modules
from face_enrollment import add_labeled_face_embedding, normalize_embedding, recognize_face
from face_detection import get_camera_feed_and_detect

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Agentic Ringbell AI",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-active {
        background-color: #e6ffe6;
        border: 1px solid #00cc00;
    }
    .status-inactive {
        background-color: #ffe6e6;
        border: 1px solid #cc0000;
    }
    .face-preview {
        border: 2px solid #ccc;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
# This block ensures all necessary session state variables are initialized when the app starts
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "face_app" not in st.session_state:
    st.session_state.face_app = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "owner_status" not in st.session_state:
    st.session_state.owner_status = "home"
if "delivery_expected" not in st.session_state:
    st.session_state.delivery_expected = False
if "agent_running" not in st.session_state:
    st.session_state.agent_running = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "vbot_graph_state" not in st.session_state:
    st.session_state.vbot_graph_state = None
if "visitor_name" not in st.session_state:
    st.session_state.visitor_name = None
if "current_phase" not in st.session_state:
    st.session_state.current_phase = "setup"  # Possible phases: setup, detection, conversation, enrollment, labeling
if "listening_for_user" not in st.session_state:
    st.session_state.listening_for_user = False  # No longer directly tied to mic, but useful as a flag if user is typing
if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False  # Flag to indicate if initial graph invocation has occurred
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "cap" not in st.session_state:  # Store the cv2.VideoCapture object
    st.session_state.cap = None
# For auto-scrolling chat
if "chat_scroll_to_bottom" not in st.session_state:
    st.session_state.chat_scroll_to_bottom = False
if "user_text_input_key" not in st.session_state:
    st.session_state.user_text_input_key = 0  # To clear text input after sending
if "unique_faces_for_labeling" not in st.session_state:
    st.session_state.unique_faces_for_labeling = []
if "current_enrollment_index" not in st.session_state:
    st.session_state.current_enrollment_index = 0


# --- System Initialization ---
@st.cache_resource
def initialize_systems():
    """
    Initializes all necessary services like FaceAnalysis and ChromaDB.
    Uses st.cache_resource to run only once, speeding up subsequent app runs.
    """
    with st.spinner("Initializing AI systems... This may contain some delay."):
        load_dotenv()  # Load environment variables, including GOOGLE_API_KEY
        try:
            # Initialize InsightFace for face analysis
            face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0)

            # Initialize ChromaDB for face embeddings storage
            CHROMA_DATA_PATH = "chroma_db_face_data"
            client = chromadb.PersistentClient(path=CHROMA_DATA_PATH, settings=Settings(anonymized_telemetry=False))
            collection = client.get_or_create_collection(
                name="face_embeddings",
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity for face embeddings
            )
            st.success("AI Systems Initialized!")
            st.session_state.initialized = True
            return face_app, collection
        except Exception as e:
            st.error(f"Critical error during system initialization: {e}")
            st.session_state.initialized = False
            return None, None


# Call system initialization on app load
st.session_state.face_app, st.session_state.collection = initialize_systems()


# --- Helper to get unique person count from ChromaDB ---
def get_unique_person_count(collection):
    """
    Counts the number of unique persons (based on 'name' metadata) in the ChromaDB collection.
    """
    try:
        all_items = collection.get(include=['metadatas'])
        metadatas = all_items.get('metadatas')
        if not metadatas:
            return 0
        names = {meta['name'] for meta in metadatas if 'name' in meta}
        return len(names)
    except Exception as e:
        st.warning(f"Could not get person count: {e}")
        return 0


# --- Face Enrollment Workflow (minor tweaks for file handling) ---
def run_streamlit_enrollment_workflow():
    """
    Handles the UI and logic for enrolling new faces into the system.
    """
    if not st.session_state.collection or not st.session_state.face_app:
        st.error("Systems not initialized. Cannot run enrollment.")
        return

    st.subheader("Enroll New Faces")
    # File uploader for selecting images
    uploaded_files = st.file_uploader(
        "Upload images for enrollment:",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="enroll_uploader"
    )

    if uploaded_files:
        st.info(f"Processing {len(uploaded_files)} image(s)...")
        session_unique_faces = []
        # Directory to save temporary face crops during enrollment
        CROPPED_FACES_DIR = "detected_face_crops_streamlit"
        if not os.path.exists(CROPPED_FACES_DIR):
            os.makedirs(CROPPED_FACES_DIR)

        SESSION_GROUPING_DISTANCE_THRESHOLD = 0.35  # Threshold for grouping similar faces within a session
        DB_SUGGESTION_RECOGNITION_THRESHOLD = 0.55

        with st.spinner("Detecting faces in uploaded images..."):
            for uploaded_file in uploaded_files:
                try:
                    # Read image bytes and decode using OpenCV
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if img is None:
                        st.warning(f"Could not read image: {uploaded_file.name}")
                        continue

                    # Detect faces in the image
                    faces = st.session_state.face_app.get(img)
                    if not faces:
                        st.warning(f"No faces detected in {uploaded_file.name}")
                        continue

                    for face_obj in faces:
                        try:
                            emb_norm_np = normalize_embedding(face_obj.embedding)
                            if emb_norm_np is None:
                                continue

                            # Check if this face is already processed in the current session
                            is_new_in_session = all(
                                1 - np.dot(emb_norm_np, np.array(uf['embedding'])) > SESSION_GROUPING_DISTANCE_THRESHOLD
                                for uf in session_unique_faces
                            )

                            if is_new_in_session:
                                # Attempt to recognize the face against the existing database for suggestions
                                name_from_db, dist_from_db = recognize_face(
                                    collection=st.session_state.collection,
                                    embedding=face_obj.embedding,
                                    threshold=DB_SUGGESTION_RECOGNITION_THRESHOLD
                                )

                                # Crop the face and save it temporarily
                                bbox = face_obj.bbox.astype(int)
                                padding = 20
                                x1, y1 = max(0, bbox[0] - padding), max(0, bbox[1] - padding)
                                x2, y2 = min(img.shape[1], bbox[2] + padding), min(img.shape[0], bbox[3] + padding)
                                
                                # Ensure valid crop dimensions
                                if x2 <= x1 or y2 <= y1:
                                    st.warning(f"Invalid face crop dimensions in {uploaded_file.name}")
                                    continue

                                crop_filename = os.path.join(CROPPED_FACES_DIR, f"session_unique_{uuid.uuid4().hex[:6]}.jpg")
                                face_crop = img[y1:y2, x1:x2]
                                
                                if face_crop.size == 0:
                                    st.warning(f"Empty face crop generated from {uploaded_file.name}")
                                    continue
                                
                                cv2.imwrite(crop_filename, face_crop)

                                session_unique_faces.append({
                                    'embedding': emb_norm_np.tolist(),
                                    'crop_path': crop_filename,
                                    'db_suggestion': name_from_db,
                                    'db_suggestion_dist': dist_from_db
                                })
                        except Exception as face_error:
                            st.warning(f"Error processing face in {uploaded_file.name}: {str(face_error)}")
                            continue

                except Exception as img_error:
                    st.error(f"Error processing image {uploaded_file.name}: {str(img_error)}")
                    continue

        st.session_state.unique_faces_for_labeling = session_unique_faces
        if session_unique_faces:
            st.success(f"Detected {len(session_unique_faces)} unique face(s) for labeling. Please proceed to label below.")
            st.session_state.current_enrollment_index = 0
            st.session_state.current_phase = "labeling"  # Transition to labeling phase
            st.rerun()  # Trigger rerun to display the labeling UI
        else:
            st.info("No new unique faces detected in uploaded images.")
            if os.path.exists(CROPPED_FACES_DIR):
                shutil.rmtree(CROPPED_FACES_DIR)


# --- UI for labeling detected faces ---
def display_labeling_ui():
    """
    Presents detected face crops to the user for labeling and adds them to the database.
    """
    if "unique_faces_for_labeling" not in st.session_state or not st.session_state.unique_faces_for_labeling:
        st.warning("No faces to label. Please upload images first.")
        st.session_state.current_phase = "setup"
        return

    idx = st.session_state.current_enrollment_index
    if idx >= len(st.session_state.unique_faces_for_labeling):
        st.success("All faces labeled!")
        st.session_state.unique_faces_for_labeling = []
        st.session_state.current_enrollment_index = 0
        st.session_state.current_phase = "setup"
        # Clean up temporary directory after all faces are labeled
        if os.path.exists("detected_face_crops_streamlit"):
            shutil.rmtree("detected_face_crops_streamlit")
        st.rerun()  # Rerun to go back to setup phase
        return

    face_to_label = st.session_state.unique_faces_for_labeling[idx]
    st.subheader(f"Label Face {idx + 1}/{len(st.session_state.unique_faces_for_labeling)}")
    
    # Display face crop with styling
    st.markdown('<div class="face-preview">', unsafe_allow_html=True)
    st.image(face_to_label['crop_path'], use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    suggestion_text = ""
    if face_to_label['db_suggestion'] and face_to_label['db_suggestion'] != 'unknown':
        suggestion_text = f" (DB Suggests: '{face_to_label['db_suggestion']}' with confidence: {1 - face_to_label['db_suggestion_dist']:.2%})"

    user_label_input = st.text_input(
        f"Who is this?{suggestion_text}",
        key=f"label_input_{idx}",
        value=face_to_label['db_suggestion'] if face_to_label['db_suggestion'] != 'unknown' else ""
    )

    col_label, col_skip = st.columns([1, 1])
    with col_label:
        if st.button("Label & Next", key=f"label_btn_{idx}"):
            if user_label_input:
                add_labeled_face_embedding(st.session_state.collection, user_label_input, face_to_label['embedding'],
                                           face_to_label['crop_path'])
                st.session_state.current_enrollment_index += 1
                st.rerun()
            else:
                st.warning("Please enter a name or click 'Skip'.")
    with col_skip:
        if st.button("Skip", key=f"skip_btn_{idx}"):
            st.session_state.current_enrollment_index += 1
            st.rerun()


# --- Main Layout Structure ---
st.title("🔔 Agentic Ringbell AI System")

# Create three columns for the main layout
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Main video and chat area
    st.subheader("Live View & Chat")
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Chat container with fixed height and scrolling
    chat_container = st.container(height=400)
    
    # Display chat messages
    with chat_container:
        for msg in st.session_state.chat_messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage) and msg.content:
                with st.chat_message("assistant"):
                    st.write(msg.content)

with col2:
    # System Status and Controls
    st.subheader("System Status")
    
    # Status indicator
    status_class = "status-active" if st.session_state.camera_active else "status-inactive"
    status_text = "Active" if st.session_state.camera_active else "Inactive"
    st.markdown(f"""
        <div class="status-box {status_class}">
            <h3>System Status: {status_text}</h3>
            <p>Current Phase: {st.session_state.current_phase.title()}</p>
            <p>Owner Status: {st.session_state.owner_status.title()}</p>
            <p>Delivery Expected: {'Yes' if st.session_state.delivery_expected else 'No'}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Control buttons
    if st.session_state.current_phase not in ["detection", "conversation"]:
        if st.button("Start Ringbell Detection", type="primary"):
            st.session_state.current_phase = "detection"
            st.session_state.agent_running = False
            st.session_state.chat_messages = []
            st.session_state.user_text_input_key = 0
            st.session_state.visitor_name = None
            st.session_state.vbot_graph_state = None
            st.session_state.listening_for_user = False
            st.session_state.conversation_started = False
            st.session_state.chat_scroll_to_bottom = True
            
            if st.session_state.cap is None or not st.session_state.cap.isOpened():
                st.session_state.cap = cv2.VideoCapture(0)
                if not st.session_state.cap.isOpened():
                    st.error("Failed to open webcam. Please check permissions.")
                    st.session_state.current_phase = "setup"
                    st.stop()
            st.session_state.camera_active = True
            st.rerun()
    
    if st.session_state.current_phase in ["detection", "conversation"]:
        if st.button("Stop System"):
            st.session_state.current_phase = "setup"
            st.session_state.agent_running = False
            st.session_state.listening_for_user = False
            st.session_state.conversation_started = False
            st.session_state.camera_active = False
            st.session_state.chat_scroll_to_bottom = False
            st.session_state.user_text_input_key = 0
            
            if st.session_state.cap is not None and st.session_state.cap.isOpened():
                st.session_state.cap.release()
                st.session_state.cap = None
            st.info("System stopped.")
            st.rerun()

with col3:
    # Face Management
    st.subheader("Face Management")
    
    # Display count of recognized persons
    num_persons = get_unique_person_count(st.session_state.collection) if st.session_state.collection else 0
    st.metric(label="Recognized Persons", value=num_persons)
    
    # Face enrollment button
    if st.button("Enroll New Faces"):
        st.session_state.current_phase = "enrollment"
        st.session_state.unique_faces_for_labeling = []
        st.session_state.current_enrollment_index = 0
        st.rerun()
    
    # Display and manage existing people
    st.markdown("---")
    st.subheader("Manage Enrolled People")
    
    if st.session_state.collection:
        try:
            # Get all items from collection
            all_items = st.session_state.collection.get(include=['metadatas', 'embeddings'])
            if all_items and all_items['metadatas']:
                # Group by name to show unique people
                people_dict = {}
                for idx, metadata in enumerate(all_items['metadatas']):
                    name = metadata.get('name', 'Unknown')
                    if name not in people_dict:
                        people_dict[name] = {
                            'count': 1,
                            'first_embedding': all_items['embeddings'][idx]
                        }
                    else:
                        people_dict[name]['count'] += 1

                # Display each person with their count and management options
                for name, data in people_dict.items():
                    with st.expander(f"👤 {name} ({data['count']} photos)"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            new_name = st.text_input(
                                "Change label:",
                                value=name,
                                key=f"rename_{name}"
                            )
                        with col2:
                            if st.button("Update", key=f"update_{name}"):
                                if new_name and new_name != name:
                                    # Update all entries with this name
                                    try:
                                        # Get all items with the old name
                                        items = st.session_state.collection.get(
                                            where={"name": name},
                                            include=['metadatas', 'embeddings']
                                        )
                                        if items and items['metadatas']:
                                            # Update each item with the new name
                                            for idx, metadata in enumerate(items['metadatas']):
                                                metadata['name'] = new_name
                                                st.session_state.collection.update(
                                                    ids=[items['ids'][idx]],
                                                    metadatas=[metadata]
                                                )
                                        st.success(f"Updated {name} to {new_name}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error updating name: {str(e)}")
                            
                            if st.button("Delete", key=f"delete_{name}"):
                                try:
                                    # Delete all entries with this name
                                    st.session_state.collection.delete(
                                        where={"name": name}
                                    )
                                    st.success(f"Deleted all entries for {name}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting entries: {str(e)}")
        except Exception as e:
            st.warning(f"Error loading enrolled people: {str(e)}")
    else:
        st.info("No face database available. Please enroll some faces first.")
    
    # Owner status controls
    st.markdown("---")
    st.subheader("Owner Settings")
    st.session_state.owner_status = st.radio(
        "Owner Status:",
        ("home", "away", "out_of_place"),
        index=("home", "away", "out_of_place").index(st.session_state.owner_status),
        help="Current status of the house owner."
    )
    st.session_state.delivery_expected = st.checkbox(
        "Delivery Expected?",
        value=st.session_state.delivery_expected,
        help="Check if a delivery is expected today."
    )

# Display enrollment or labeling UI based on current phase
if st.session_state.current_phase == "enrollment":
    run_streamlit_enrollment_workflow()
elif st.session_state.current_phase == "labeling":
    display_labeling_ui()

# --- Main Application Logic based on current_phase ---

if st.session_state.current_phase == "detection":
    # Use st.status to provide persistent feedback during detection
    with st.status("Initializing Camera and Looking for Visitors...", expanded=True,
                   state="running") as detection_status:
        video_output = video_placeholder.empty()  # Placeholder for live video feed
        current_status_text = status_placeholder.empty()  # Placeholder for status messages

        # Check if camera is active and opened
        if not st.session_state.camera_active or st.session_state.cap is None or not st.session_state.cap.isOpened():
            try:
                st.session_state.cap = cv2.VideoCapture(0)
                if not st.session_state.cap.isOpened():
                    detection_status.update(label="Camera Not Active or Failed to Open", state="error", expanded=True)
                    current_status_text.error(
                        "Camera is not active or failed to open. Check permissions and ensure no other app is using it.")
                    st.session_state.current_phase = "setup"
                    st.session_state.camera_active = False
                    st.stop()
                st.session_state.camera_active = True
            except Exception as e:
                detection_status.update(label="Camera Error", state="error", expanded=True)
                current_status_text.error(f"Error initializing camera: {str(e)}")
                st.session_state.current_phase = "setup"
                st.session_state.camera_active = False
                st.stop()

        # Get a single frame from the camera and perform detection
        try:
            frame_data, detected_name, status_message = get_camera_feed_and_detect(
                st.session_state.face_app, st.session_state.collection, st.session_state.cap
            )

            if frame_data:
                video_output.image(frame_data, channels="BGR", use_container_width=True)
                current_status_text.text(status_message)
                detection_status.update(label=f"Camera Active: {status_message}", state="running", expanded=True)

                if detected_name:
                    st.session_state.visitor_name = detected_name
                    st.session_state.current_phase = "conversation"
                    st.session_state.agent_running = True
                    st.session_state.conversation_started = False
                    st.session_state.chat_scroll_to_bottom = True
                    detection_status.update(label=f"Visitor identified: {detected_name}. Starting VBot conversation.",
                                            state="complete", expanded=False)
                    # Release camera as detection phase is ending
                    if st.session_state.cap is not None and st.session_state.cap.isOpened():
                        st.session_state.cap.release()
                        st.session_state.cap = None
                    st.session_state.camera_active = False
                    st.rerun()
                else:
                    # If no face detected, wait 5 seconds before next attempt
                    time.sleep(5)
                    st.rerun()
            else:
                # Handle cases where frame_data is None
                detection_status.update(label=f"Detection Error: {status_message}", state="error", expanded=True)
                current_status_text.error(status_message)
                # Don't stop the detection phase, just wait and retry
                time.sleep(5)
                st.rerun()

        except Exception as e:
            detection_status.update(label="Detection Error", state="error", expanded=True)
            current_status_text.error(f"Error during detection: {str(e)}")
            # Don't stop the detection phase, just wait and retry
            time.sleep(5)
            st.rerun()

elif st.session_state.current_phase == "conversation":
    # Initialize conversation if not started
    if not st.session_state.conversation_started:
        st.session_state.chat_messages = []
        st.session_state.user_text_input_key = 0

        # Add initial human message
        initial_human_message = HumanMessage(content=f"My name is {st.session_state.visitor_name}")
        st.session_state.chat_messages.append(initial_human_message)

        # Display the initial human message immediately
        with chat_container:
            with st.chat_message("user"):
                st.write(initial_human_message.content)

        # Initialize graph state
        initial_state = SessionState(
            messages=[initial_human_message],
            authenticated=False,
            finished=False,
            owner_status=st.session_state.owner_status,
            delivery_expected=st.session_state.delivery_expected,
            frequency_updated=False
        )
        st.session_state.vbot_graph_state = initial_state
        st.session_state.agent_running = True
        st.session_state.conversation_started = True
        st.session_state.chat_scroll_to_bottom = True

        try:
            with status_placeholder.container():
                with st.spinner("VBot thinking..."):
                    new_graph_state = graph.invoke(st.session_state.vbot_graph_state)

            st.session_state.vbot_graph_state = new_graph_state
            # Only add non-None messages to chat history
            new_messages_from_graph = [
                msg for msg in new_graph_state["messages"]
                if msg not in st.session_state.chat_messages and 
                (not isinstance(msg, AIMessage) or msg.content is not None)
            ]
            st.session_state.chat_messages.extend(new_messages_from_graph)

            # Display new messages
            with chat_container:
                for msg in new_messages_from_graph:
                    if isinstance(msg, AIMessage) and msg.content:
                        with st.chat_message("assistant"):
                            st.write(msg.content)
                            speak_text(msg.content)

            st.rerun()
        except Exception as e:
            st.error(f"Error during initial agent invocation: {e}")
            st.session_state.agent_running = False
            st.session_state.current_phase = "setup"
            st.stop()

    # Handle ongoing conversation
    current_graph_state = st.session_state.vbot_graph_state
    if current_graph_state is None:
        st.error("VBot graph state is unexpectedly None during ongoing conversation. Re-starting.")
        st.session_state.current_phase = "setup"
        st.session_state.agent_running = False
        st.session_state.conversation_started = False
        st.stop()

    # Check if conversation is finished
    if current_graph_state.get("finished", False):
        st.info("VBot conversation finished. Returning to detection mode in 5 seconds...")
        st.session_state.agent_running = False
        st.session_state.current_phase = "detection"
        st.session_state.conversation_started = False
        st.session_state.chat_scroll_to_bottom = True
        
        if st.session_state.vbot_graph_state and st.session_state.vbot_graph_state["messages"]:
            final_msg = st.session_state.vbot_graph_state["messages"][-1]
            if isinstance(final_msg, AIMessage) and final_msg.content and final_msg not in st.session_state.chat_messages:
                st.session_state.chat_messages.append(final_msg)
        
        # Clear chat messages for next interaction
        st.session_state.chat_messages = []
        st.session_state.user_text_input_key = 0
        st.session_state.visitor_name = None
        st.session_state.vbot_graph_state = None
        
        # Wait 5 seconds before restarting detection
        time.sleep(5)
        
        # Reinitialize camera for detection
        try:
            st.session_state.cap = cv2.VideoCapture(0)
            if not st.session_state.cap.isOpened():
                st.error("Failed to reinitialize camera. Please check permissions.")
                st.session_state.current_phase = "setup"
                st.stop()
            st.session_state.camera_active = True
            st.rerun()
        except Exception as e:
            st.error(f"Error reinitializing camera: {str(e)}")
            st.session_state.current_phase = "setup"
            st.stop()

    # Handle ongoing conversation flow
    elif st.session_state.agent_running:
        # Get the next possible nodes from the graph
        next_node_options = graph.get_next_steps(current_graph_state)
        print(f"\nDEBUG: Next node options: {next_node_options}")  # Debug print

        # Process the graph step first to get any new messages
        try:
            with status_placeholder.container():
                with st.spinner("VBot thinking..."):
                    new_graph_state = graph.invoke(current_graph_state)

            st.session_state.vbot_graph_state = new_graph_state
            # Only add non-None messages to chat history
            new_messages_from_graph = [
                msg for msg in new_graph_state["messages"]
                if msg not in st.session_state.chat_messages and 
                (not isinstance(msg, AIMessage) or msg.content is not None)
            ]
            st.session_state.chat_messages.extend(new_messages_from_graph)
            st.session_state.chat_scroll_to_bottom = True

            # Display all messages in chat container
            with chat_container:
                for msg in st.session_state.chat_messages:
                    if isinstance(msg, HumanMessage):
                        with st.chat_message("user"):
                            st.write(msg.content)
                    elif isinstance(msg, AIMessage) and msg.content:
                        with st.chat_message("assistant"):
                            st.write(msg.content)
                            speak_text(msg.content)

            # Check if we're in the human node
            is_human_turn = "human" in next_node_options
            print(f"DEBUG: Is human turn: {is_human_turn}")  # Debug print

            if is_human_turn:
                print("DEBUG: Waiting for human input")  # Debug print
                
                # Add text input at the bottom
                user_text_input = st.text_input(
                    "Your response:",
                    key=f"user_text_input_{st.session_state.user_text_input_key}"
                )

                if st.button("Send", key="send_text_button"):
                    if user_text_input:
                        print(f"DEBUG: User input received: {user_text_input}")  # Debug print
                        user_message = HumanMessage(content=user_text_input)
                        current_graph_state["messages"].append(user_message)
                        st.session_state.chat_messages.append(user_message)
                        st.session_state.chat_scroll_to_bottom = True
                        st.session_state.user_text_input_key += 1
                        st.rerun()
                    else:
                        st.warning("Please enter some text.")
                else:
                    # If no input has been provided yet, don't proceed with the graph
                    print("DEBUG: Waiting for user to click Send")  # Debug print
                    st.stop()
            else:
                # If it's not human's turn, continue processing
                print("DEBUG: Processing next graph step")  # Debug print
                st.rerun()

        except Exception as e:
            st.error(f"Error during agent processing: {e}")
            st.session_state.agent_running = False
            st.session_state.current_phase = "setup"
            st.stop()






