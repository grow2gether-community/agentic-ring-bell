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
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .status-box h3 {
        font-size: 1rem;
        margin: 0 0 0.5rem 0;
    }
    .status-box p {
        margin: 0.2rem 0;
        font-size: 0.9rem;
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
    .stSubheader {
        font-size: 1.2rem !important;
        margin-bottom: 0.5rem !important;
    }
    .enrolled-people-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ccc;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
    /* Style for the expander container */
    .streamlit-expanderHeader {
        font-size: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
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
    st.session_state.current_phase = "setup"
if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "chat_scroll_to_bottom" not in st.session_state:
    st.session_state.chat_scroll_to_bottom = False
if "user_text_input_key" not in st.session_state:
    st.session_state.user_text_input_key = 0
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "awaiting_human_input_ui" not in st.session_state:
    st.session_state.awaiting_human_input_ui = False


# --- System Initialization ---
@st.cache_resource
def initialize_systems():
    with st.spinner("Initializing AI systems... This may contain some delay."):
        load_dotenv()
        try:
            face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0)

            CHROMA_DATA_PATH = "chroma_db_face_data"
            client = chromadb.PersistentClient(path=CHROMA_DATA_PATH, settings=Settings(anonymized_telemetry=False))
            collection = client.get_or_create_collection(
                name="face_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            st.success("AI Systems Initialized!")
            st.session_state.initialized = True
            return face_app, collection
        except Exception as e:
            st.error(f"Critical error during system initialization: {e}")
            st.session_state.initialized = False
            return None, None

st.session_state.face_app, st.session_state.collection = initialize_systems()

# --- Helper to get unique person count from ChromaDB ---
def get_unique_person_count(collection):
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

# --- Face Enrollment Workflow ---
def run_streamlit_enrollment_workflow():
    if not st.session_state.collection or not st.session_state.face_app:
        st.error("Systems not initialized. Cannot run enrollment.")
        return

    st.subheader("Enroll New Faces")
    uploaded_files = st.file_uploader(
        "Upload images for enrollment:",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="enroll_uploader"
    )

    if uploaded_files:
        st.info(f"Processing {len(uploaded_files)} image(s)...")
        session_unique_faces = []
        CROPPED_FACES_DIR = "detected_face_crops_streamlit"
        if not os.path.exists(CROPPED_FACES_DIR):
            os.makedirs(CROPPED_FACES_DIR)

        SESSION_GROUPING_DISTANCE_THRESHOLD = 0.35
        DB_SUGGESTION_RECOGNITION_THRESHOLD = 0.55

        with st.spinner("Detecting faces in uploaded images..."):
            for uploaded_file in uploaded_files:
                try:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if img is None:
                        st.warning(f"Could not read image: {uploaded_file.name}")
                        continue

                    faces = st.session_state.face_app.get(img)
                    if not faces:
                        st.warning(f"No faces detected in {uploaded_file.name}")
                        continue

                    for face_obj in faces:
                        try:
                            emb_norm_np = normalize_embedding(face_obj.embedding)
                            if emb_norm_np is None:
                                continue

                            is_new_in_session = all(
                                1 - np.dot(emb_norm_np, np.array(uf['embedding'])) > SESSION_GROUPING_DISTANCE_THRESHOLD
                                for uf in session_unique_faces
                            )

                            if is_new_in_session:
                                name_from_db, dist_from_db = recognize_face(
                                    collection=st.session_state.collection,
                                    embedding=face_obj.embedding,
                                    threshold=DB_SUGGESTION_RECOGNITION_THRESHOLD
                                )

                                bbox = face_obj.bbox.astype(int)
                                padding = 20
                                x1, y1 = max(0, bbox[0] - padding), max(0, bbox[1] - padding)
                                x2, y2 = min(img.shape[1], bbox[2] + padding), min(img.shape[0], bbox[3] + padding)
                                
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
            st.session_state.current_phase = "labeling"
            st.rerun()
        else:
            st.info("No new unique faces detected in uploaded images.")
            if os.path.exists(CROPPED_FACES_DIR):
                shutil.rmtree(CROPPED_FACES_DIR)

# --- UI for labeling detected faces ---
def display_labeling_ui():
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
        if os.path.exists("detected_face_crops_streamlit"):
            shutil.rmtree("detected_face_crops_streamlit")
        st.rerun()
        return

    face_to_label = st.session_state.unique_faces_for_labeling[idx]
    st.subheader(f"Label Face {idx + 1}/{len(st.session_state.unique_faces_for_labeling)}")
    
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

# --- Core Agent Invocation Logic ---
def _invoke_agent_single_turn():
    """
    Function to encapsulate a single turn of the agent's processing.
    This function will be called explicitly when needed, and will perform only ONE graph.invoke() call.
    """
    current_graph_state = st.session_state.vbot_graph_state
    
    # Initialize graph state for a new conversation if not already done
    if current_graph_state is None:
        current_graph_state = SessionState(
            messages=[],
            authenticated=False,
            finished=False,
            owner_status=st.session_state.owner_status,
            delivery_expected=st.session_state.delivery_expected,
            frequency_updated=False
        )
        st.session_state.vbot_graph_state = current_graph_state

    # Add initial human message for a new conversation turn if starting
    # This happens only ONCE per new conversation (when conversation_started is False)
    if not st.session_state.conversation_started:
        print("DEBUG app.py: Starting new conversation. Adding initial human message.")
        initial_human_message = HumanMessage(content=f"My name is {st.session_state.visitor_name}")
        st.session_state.chat_messages.append(initial_human_message)
        st.session_state.vbot_graph_state["messages"].append(initial_human_message)
        st.session_state.conversation_started = True
        st.session_state.chat_scroll_to_bottom = True

    print("DEBUG app.py: Invoking graph for a single agent turn...")
    try:
        # Pass the current graph state to the graph
        new_graph_state = graph.invoke(st.session_state.vbot_graph_state)
        
        # Update the session state with the new graph state returned by the invocation
        st.session_state.vbot_graph_state = new_graph_state
        
        # Append only new messages (HumanMessage or AIMessage) from the graph invocation to chat history
        for msg in new_graph_state["messages"]:
            if msg not in st.session_state.chat_messages and isinstance(msg, (AIMessage, HumanMessage)):
                st.session_state.chat_messages.append(msg)
        
        # IMPORTANT: NO st.rerun() HERE. This function performs one invoke and returns.
        # The main app loop will handle the display update.

    except Exception as e:
        st.error(f"Error during agent interaction: {e}")
        st.session_state.agent_running = False
        st.session_state.current_phase = "setup"
        # Let Streamlit handle reruns; do not use st.stop() directly here as it can disrupt flow.


# --- Main Layout Structure ---
st.title("🔔 Agentic Ringbell AI System")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("System Status")
    
    status_class = "status-active" if st.session_state.camera_active else "status-inactive"
    status_text = "Active" if st.session_state.camera_active else "Inactive"
    st.markdown(f"""
        <div class="status-box {status_class}">
            <h3>System Status: {status_text}</h3>
            <p>Phase: {st.session_state.current_phase.title()}</p>
            <p>Owner: {st.session_state.owner_status.title()}</p>
            <p>Delivery: {'Yes' if st.session_state.delivery_expected else 'No'}</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.current_phase not in ["detection", "conversation"]:
        if st.button("Start Ringbell Detection", type="primary"):
            st.session_state.current_phase = "detection"
            st.session_state.agent_running = False
            st.session_state.chat_messages = []
            st.session_state.user_text_input_key = 0
            st.session_state.visitor_name = None
            st.session_state.vbot_graph_state = None # Ensure a fresh start for graph state
            st.session_state.conversation_started = False
            st.session_state.chat_scroll_to_bottom = True
            st.session_state.awaiting_human_input_ui = False
            st.session_state.user_input = ""
            
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
            st.session_state.conversation_started = False
            st.session_state.camera_active = False
            st.session_state.chat_scroll_to_bottom = False
            st.session_state.user_text_input_key = 0
            st.session_state.user_input = ""
            st.session_state.awaiting_human_input_ui = False
            st.session_state.chat_messages = []  # Clear chat messages
            
            if st.session_state.cap is not None and st.session_state.cap.isOpened():
                st.session_state.cap.release()
            st.session_state.cap = None
            st.info("System stopped.")
            st.rerun()

        st.markdown("---")
    st.subheader("Owner Settings")
    
    # Create a form for owner settings to ensure immediate updates
    with st.form("owner_settings_form"):
        new_owner_status = st.radio(
            "Owner Status:",
            ("home", "away", "out_of_place"),
            index=("home", "away", "out_of_place").index(st.session_state.owner_status),
            help="Current status of the house owner."
        )
        new_delivery_expected = st.checkbox(
            "Delivery Expected?",
            value=st.session_state.delivery_expected,
            help="Check if a delivery is expected today."
        )

        # Update settings when form is submitted
        if st.form_submit_button("Update Settings"):
            st.session_state.owner_status = new_owner_status
            st.session_state.delivery_expected = new_delivery_expected
            st.rerun()

with col2:
    st.subheader("Live View & Chat")
    status_placeholder = st.empty()
    
    chat_container = st.container(height=400)

    # Display chat messages and webcam feed in the same container
    with chat_container:
        # Clear previous content if in detection phase
        if st.session_state.current_phase == "detection":
            st.empty()  # Clear previous content
            
            # Display webcam feed if camera is active
            if st.session_state.cap is not None and st.session_state.cap.isOpened():
                try:
                    frame_data, detected_name, status_message = get_camera_feed_and_detect(
                        st.session_state.face_app, st.session_state.collection, st.session_state.cap
                    )
                    if frame_data is not None:
                        st.image(frame_data, channels="BGR", use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying webcam feed: {str(e)}")

        # Display chat messages only in conversation phase
        if st.session_state.current_phase == "conversation":
            for msg in st.session_state.chat_messages:
                if isinstance(msg, HumanMessage):
                    with st.chat_message("user"):
                        st.write(msg.content)
                elif isinstance(msg, AIMessage) and msg.content:
                    with st.chat_message("assistant"):
                        st.write(msg.content)
                        # Speak only if it's the latest AI message and not yet spoken
                        if msg == st.session_state.chat_messages[-1] and "spoken_flag" not in msg.additional_kwargs:
                            speak_text(msg.content)
                            msg.additional_kwargs["spoken_flag"] = True

with col3:
    st.subheader("Face Management")
    num_persons = get_unique_person_count(st.session_state.collection) if st.session_state.collection else 0
    st.metric(label="Recognized Persons", value=num_persons)
    
    # Face Enrollment Section
    st.markdown("---")
    st.subheader("Enroll New Faces")
    
    # Create a container for face enrollment
    with st.container():
        # Show labeling UI if we have faces to label
        if st.session_state.get("current_phase") == "labeling" and st.session_state.get("unique_faces_for_labeling"):
            idx = st.session_state.current_enrollment_index
            if idx < len(st.session_state.unique_faces_for_labeling):
                face_to_label = st.session_state.unique_faces_for_labeling[idx]
                st.subheader(f"Label Face {idx + 1}/{len(st.session_state.unique_faces_for_labeling)}")
                
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
            else:
                st.success("All faces labeled!")
                st.session_state.unique_faces_for_labeling = []
                st.session_state.current_enrollment_index = 0
                st.session_state.current_phase = "setup"
                if os.path.exists("detected_face_crops_streamlit"):
                    shutil.rmtree("detected_face_crops_streamlit")
                st.rerun()
        else:
            # Show file uploader if we're not in labeling phase
            uploaded_files = st.file_uploader(
                "Upload images for enrollment:",
                type=["jpg", "jpeg", "png", "bmp"],
                accept_multiple_files=True,
                key="enroll_uploader"
            )

            if uploaded_files:
                st.info(f"Processing {len(uploaded_files)} image(s)...")
                session_unique_faces = []
                CROPPED_FACES_DIR = "detected_face_crops_streamlit"
                if not os.path.exists(CROPPED_FACES_DIR):
                    os.makedirs(CROPPED_FACES_DIR)

                SESSION_GROUPING_DISTANCE_THRESHOLD = 0.35
                DB_SUGGESTION_RECOGNITION_THRESHOLD = 0.55

                with st.spinner("Detecting faces in uploaded images..."):
                    for uploaded_file in uploaded_files:
                        try:
                            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                            if img is None:
                                st.warning(f"Could not read image: {uploaded_file.name}")
                                continue

                            faces = st.session_state.face_app.get(img)
                            if not faces:
                                st.warning(f"No faces detected in {uploaded_file.name}")
                                continue

                            for face_obj in faces:
                                try:
                                    emb_norm_np = normalize_embedding(face_obj.embedding)
                                    if emb_norm_np is None:
                                        continue

                                    is_new_in_session = all(
                                        1 - np.dot(emb_norm_np, np.array(uf['embedding'])) > SESSION_GROUPING_DISTANCE_THRESHOLD
                                        for uf in session_unique_faces
                                    )

                                    if is_new_in_session:
                                        name_from_db, dist_from_db = recognize_face(
                                            collection=st.session_state.collection,
                                            embedding=face_obj.embedding,
                                            threshold=DB_SUGGESTION_RECOGNITION_THRESHOLD
                                        )

                                        bbox = face_obj.bbox.astype(int)
                                        padding = 20
                                        x1, y1 = max(0, bbox[0] - padding), max(0, bbox[1] - padding)
                                        x2, y2 = min(img.shape[1], bbox[2] + padding), min(img.shape[0], bbox[3] + padding)
                                        
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
                    st.success(f"Detected {len(session_unique_faces)} unique face(s) for labeling.")
                    st.session_state.current_enrollment_index = 0
                    st.session_state.current_phase = "labeling"
                    st.rerun()
                else:
                    st.info("No new unique faces detected in uploaded images.")
                    if os.path.exists(CROPPED_FACES_DIR):
                        shutil.rmtree(CROPPED_FACES_DIR)

    # Manage Enrolled People Section
    st.markdown("---")
    if st.button("Manage Enrolled People"):
        st.session_state.show_enrolled_people = not st.session_state.get("show_enrolled_people", False)
        st.rerun()
    
    if st.session_state.get("show_enrolled_people", False):
        st.subheader("Enrolled People")
        with st.container():
            st.markdown('<div class="enrolled-people-container">', unsafe_allow_html=True)
            if st.session_state.collection:
                try:
                    all_items = st.session_state.collection.get(include=['metadatas', 'embeddings'])
                    if all_items and all_items['metadatas']:
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

                        expander_container = st.container()
                        with expander_container:
                            for name, data in people_dict.items():
                                with st.expander(f"👤 {name} ({data['count']} photos)"):
                                    col1, col2 = st.columns([2, 1])
                                    with col1:
                                        new_name = st.text_input(
                                            "Change label:",
                                            value=name,
                                            key=f"rename_{name}"
                                        )
                                    with col2:
                                        if st.button("Update", key=f"update_{name}", use_container_width=True):
                                            if new_name and new_name != name:
                                                try:
                                                    items = st.session_state.collection.get(
                                                        where={"name": name},
                                                        include=['metadatas', 'embeddings']
                                                    )
                                                    if items and items['metadatas']:
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
                                        
                                        if st.button("Delete", key=f"delete_{name}", use_container_width=True):
                                            try:
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
            st.markdown('</div>', unsafe_allow_html=True)

# --- Main Application Logic based on current_phase ---
if st.session_state.current_phase == "detection":
    with st.status("Initializing Camera and Looking for Visitors...", expanded=True, state="running") as detection_status:
        current_status_text = status_placeholder.empty()

        if not st.session_state.camera_active or st.session_state.cap is None or not st.session_state.cap.isOpened():
            try:
                st.session_state.cap = cv2.VideoCapture(0)
                if not st.session_state.cap.isOpened():
                    detection_status.update(label="Camera Not Active or Failed to Open", state="error", expanded=True)
                    current_status_text.error("Camera is not active or failed to open. Check permissions and ensure no other app is using it.")
                    st.session_state.current_phase = "setup"
                    st.stop()
                st.session_state.camera_active = True
            except Exception as e:
                detection_status.update(label="Camera Error", state="error", expanded=True)
                current_status_text.error(f"Error initializing camera: {str(e)}")
                st.session_state.current_phase = "setup"
                st.stop()

        try:
            frame_data, detected_name, status_message = get_camera_feed_and_detect(
                st.session_state.face_app, st.session_state.collection, st.session_state.cap
            )

            if frame_data is not None:
                current_status_text.text(status_message)
                detection_status.update(label=f"Camera Active: {status_message}", state="running", expanded=True)

                if detected_name:
                    st.session_state.visitor_name = detected_name
                    st.session_state.current_phase = "conversation"
                    st.session_state.agent_running = True
                    st.session_state.conversation_started = False # Flag for new conversation
                    st.session_state.chat_messages = [] # Clear chat for new convo
                    st.session_state.chat_scroll_to_bottom = True
                    detection_status.update(label=f"Visitor identified: {detected_name}. Starting VBot conversation.",
                                            state="complete", expanded=False)
                    if st.session_state.cap is not None and st.session_state.cap.isOpened():
                        st.session_state.cap.release()
                    st.session_state.cap = None
                    st.session_state.camera_active = False
                    st.rerun() # Trigger rerun to enter conversation phase
                else:
                    time.sleep(5)  # Changed to 5 seconds as requested
                    st.rerun()
            else:
                detection_status.update(label=f"Detection Error: {status_message}", state="error", expanded=True)
                current_status_text.error(status_message)
                time.sleep(5)  # Changed to 5 seconds as requested
                st.rerun()

        except Exception as e:
            detection_status.update(label="Detection Error", state="error", expanded=True)
            current_status_text.error(f"Error during detection: {str(e)}")
            time.sleep(5)  # Changed to 5 seconds as requested
            st.rerun()

elif st.session_state.current_phase == "conversation":
    # Ensure current_graph_state is available for this rerun
    if st.session_state.vbot_graph_state is None:
        # This state should be initialized if coming from detection phase, but as a safeguard.
        st.session_state.vbot_graph_state = SessionState(
            messages=[],
            authenticated=False,
            finished=False,
            owner_status=st.session_state.owner_status,
            delivery_expected=st.session_state.delivery_expected,
            frequency_updated=False
        )
    
    # Check for conversation end condition first (if the agent decided to finish)
    if st.session_state.vbot_graph_state.get("finished", False):
        st.info("VBot conversation finished. Returning to detection mode in 5 seconds...")
        if st.session_state.vbot_graph_state["messages"]:
            final_msg = st.session_state.vbot_graph_state["messages"][-1]
            if isinstance(final_msg, AIMessage) and final_msg.content and final_msg not in st.session_state.chat_messages:
                st.session_state.chat_messages.append(final_msg)
        
        # Reset all conversation-related states
        st.session_state.agent_running = False
        st.session_state.conversation_started = False
        st.session_state.chat_scroll_to_bottom = True
        st.session_state.chat_messages = []
        st.session_state.user_text_input_key = 0
        st.session_state.visitor_name = None
        st.session_state.vbot_graph_state = None
        
        time.sleep(5)
        st.session_state.current_phase = "detection"
        
        try:
            st.session_state.cap = cv2.VideoCapture(0)
            if not st.session_state.cap.isOpened():
                st.error("Failed to reinitialize camera for detection. Please check permissions.")
                st.session_state.current_phase = "setup"
                st.stop()
            st.session_state.camera_active = True
            st.rerun()
        except Exception as e:
            st.error(f"Error reinitializing camera: {str(e)}")
            st.session_state.current_phase = "setup"
            st.stop()

    # If conversation is NOT finished, proceed with turn-based logic
    else:
        # This condition ensures _invoke_agent_single_turn() is called ONLY once
        # when a new conversation starts
        if not st.session_state.conversation_started:
            _invoke_agent_single_turn()
            st.rerun() # Trigger a rerun to display the result of the single turn.

# Auto-scroll chat to bottom if chat_scroll_to_bottom is True
if st.session_state.chat_scroll_to_bottom:
    st.session_state.chat_scroll_to_bottom = False
    st.markdown(
        """
        <script>
            var chat_container = document.querySelector('[data-testid="stVerticalBlock"]');
            if (chat_container) {
                setTimeout(function() {
                    chat_container.scrollTop = chat_container.scrollHeight;
                }, 100); 
            }
        </script>
        """,
        unsafe_allow_html=True
    )