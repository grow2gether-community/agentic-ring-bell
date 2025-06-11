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
    layout="wide"
)

st.title("🔔 Agentic Ringbell AI System")

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
        DB_SUGGESTION_RECOGNITION_THRESHOLD = 0.55  # Threshold for suggesting names from the existing DB

        with st.spinner("Detecting faces in uploaded images..."):
            for uploaded_file in uploaded_files:
                # Read image bytes and decode using OpenCV
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    st.warning(f"Could not read image: {uploaded_file.name}")
                    continue

                # Detect faces in the image
                for face_obj in st.session_state.face_app.get(img):
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
                        x2, y2 = min(img.shape[1], bbox[2] + padding), min(img.shape[0], img.shape[3] + padding)
                        crop_filename = os.path.join(CROPPED_FACES_DIR, f"session_unique_{uuid.uuid4().hex[:6]}.jpg")
                        cv2.imwrite(crop_filename, img[y1:y2, x1:x2])

                        session_unique_faces.append({
                            'embedding': emb_norm_np.tolist(),
                            'crop_path': crop_filename,
                            'db_suggestion': name_from_db,
                            'db_suggestion_dist': dist_from_db
                        })
        st.session_state.unique_faces_for_labeling = session_unique_faces
        if session_unique_faces:
            st.success(
                f"Detected {len(session_unique_faces)} unique face(s) for labeling. Please proceed to label below.")
            st.session_state.current_enrollment_index = 0
            st.session_state.current_phase = "labeling"  # Transition to labeling phase
            st.rerun()  # Trigger rerun to display the labeling UI
        else:
            st.info("No new unique faces detected in uploaded images.")


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
    st.image(face_to_label['crop_path'], caption="Face to Label",
             use_container_width=True)  # Changed to use_container_width

    suggestion_text = ""
    if face_to_label['db_suggestion'] and face_to_label['db_suggestion'] != 'unknown':
        suggestion_text = f" (DB Suggests: '{face_to_label['db_suggestion']}' with dist: {face_to_label['db_suggestion_dist']:.3f})"

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
col1, col2 = st.columns([1.5, 1])  # Left column for video and chat, Right column for controls

with col1:
    st.subheader("Live View & VBot Interaction")
    # Placeholders for dynamic content, updated during execution
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    chat_container = st.container(height=400)  # Define chat_container

    # Display chat messages in the chat window (this loop runs on every rerun)
    # This initial loop iterates through all messages in chat_messages
    with chat_container:
        for msg in st.session_state.chat_messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                # Only display content of AIMessage, not internal tool calls or empty messages
                if msg.content:  # Only write if there is actual content
                    with st.chat_message("assistant"):
                        st.write(msg.content)
                # Tool calls and Tool messages are internal processing, not conversational output for the user
                # We do not want to display these in the primary chat window for the user
            # elif isinstance(msg, ToolMessage):
            #     # This block is commented out as requested for cleaner UI output
            #     pass

    # Scroll to bottom after all messages are displayed
    if st.session_state.chat_scroll_to_bottom:
        st.markdown("<script>window.scrollBy(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
        st.session_state.chat_scroll_to_bottom = False  # Reset flag

    # Dynamic button for user input during conversation phase
    if st.session_state.current_phase == "conversation" and st.session_state.agent_running:
        # Check if conversation is NOT finished AND the last message is an AIMessage (AI has spoken and expects human)
        if st.session_state.vbot_graph_state is not None and \
                not st.session_state.vbot_graph_state.get("finished", False) and \
                st.session_state.chat_messages and \
                isinstance(st.session_state.chat_messages[-1], AIMessage):  # Only show if last message is an AIMessage

            # Retrieve the current graph state to check next steps
            current_graph_state = st.session_state.vbot_graph_state
            next_node_options = graph.get_next_steps(current_graph_state)

            # CRITICAL CHECK: If the graph is explicitly routed to 'human', show the text input and send button.
            if "human" in next_node_options:
                # Text input for user's response
                user_text_input = st.text_input(
                    "Your response:",
                    key=f"user_text_input_{st.session_state.user_text_input_key}",  # Unique key for text input
                    on_change=None  # Removed direct on_change, will use button click
                )

                # Send button
                if st.button("Send", key="send_text_button"):
                    if user_text_input:
                        # Append user message to graph state and chat history
                        user_message = HumanMessage(content=user_text_input)
                        current_graph_state["messages"].append(user_message)  # Append to graph state
                        st.session_state.chat_messages.append(user_message)  # Append for UI display
                        st.session_state.chat_scroll_to_bottom = True  # Scroll to new message

                        # Clear the text input field
                        st.session_state.user_text_input_key += 1  # Increment key to clear input

                        # Invoke graph to continue conversation
                        try:
                            with status_placeholder.container():
                                with st.spinner("VBot thinking..."):
                                    new_graph_state = graph.invoke(current_graph_state)

                            st.session_state.vbot_graph_state = new_graph_state
                            new_messages_from_graph = [
                                msg for msg in new_graph_state["messages"]
                                if msg not in st.session_state.chat_messages
                            ]
                            st.session_state.chat_messages.extend(new_messages_from_graph)
                            st.session_state.chat_scroll_to_bottom = True  # Prepare to scroll chat

                            # Explicitly display new AI messages to chat_container
                            with chat_container:
                                for msg in new_messages_from_graph:
                                    if isinstance(msg, AIMessage):
                                        if msg.content:
                                            with st.chat_message("assistant"):
                                                st.write(msg.content)
                                            speak_text(msg.content)

                            st.rerun()  # Trigger rerun to update UI
                        except Exception as e:
                            st.error(f"Error during agent interaction: {e}")
                            st.session_state.agent_running = False
                            st.session_state.current_phase = "setup"
                            st.stop()
                    else:
                        st.warning("Please enter some text.")
            else:
                # If next_node is NOT 'human', the graph should auto-progress (tool call or next AI turn)
                # This state is handled by the auto-processing block below, so no input field here.
                pass
        elif st.session_state.vbot_graph_state is not None and st.session_state.vbot_graph_state.get("finished", False):
            # If conversation is finished, hide interaction elements
            pass

with col2:  # Sidebar for controls
    with st.sidebar:
        st.subheader("System Controls")

        st.markdown("---")
        st.subheader("Owner Status & Delivery")
        # Radio buttons for owner status
        st.session_state.owner_status = st.radio(
            "Owner Status:",
            ("home", "away", "out_of_place"),
            index=("home", "away", "out_of_place").index(st.session_state.owner_status),
            help="Current status of the house owner."
        )
        # Checkbox for delivery expectation
        st.session_state.delivery_expected = st.checkbox(
            "Delivery Expected?",
            value=st.session_state.delivery_expected,
            help="Check if a delivery is expected today."
        )

        st.markdown("---")
        # Display count of recognized persons in the database
        num_persons = get_unique_person_count(st.session_state.collection) if st.session_state.collection else 0
        st.metric(label="Recognized Persons in DB", value=num_persons)

        # Button to initiate face enrollment workflow
        if st.button("Enroll New Faces", key="enroll_faces_button"):
            st.session_state.current_phase = "enrollment"
            st.session_state.unique_faces_for_labeling = []  # Clear previous enrollment state
            st.session_state.current_enrollment_index = 0
            st.rerun()

        # Display enrollment or labeling UI based on current phase
        if st.session_state.current_phase == "enrollment":
            run_streamlit_enrollment_workflow()
        elif st.session_state.current_phase == "labeling":
            display_labeling_ui()

        st.markdown("---")
        # Control buttons for main system (Start/Stop Detection)
        if st.session_state.current_phase not in ["detection", "conversation"]:  # Only show Start if not active
            if st.button("Start Ringbell Detection", key="start_detection_button", type="primary"):
                st.session_state.current_phase = "detection"
                st.session_state.agent_running = False
                st.session_state.chat_messages = []
                st.session_state.user_text_input_key = 0  # Reset text input key
                st.session_state.visitor_name = None
                st.session_state.vbot_graph_state = None
                st.session_state.listening_for_user = False  # Not listening for speech
                st.session_state.conversation_started = False  # Reset this flag for a new conversation cycle
                st.session_state.chat_scroll_to_bottom = True  # Reset scroll
                # Attempt to open the webcam
                if st.session_state.cap is None or not st.session_state.cap.isOpened():
                    st.session_state.cap = cv2.VideoCapture(0)  # Index 0 for default webcam
                    if not st.session_state.cap.isOpened():
                        st.error("Failed to open webcam. Ensure no other app is using it and permissions are granted.")
                        st.session_state.current_phase = "setup"  # Revert to setup on camera failure
                        st.stop()  # Halt current execution
                st.session_state.camera_active = True
                st.rerun()  # Trigger rerun to start the detection loop

        if st.session_state.current_phase in ["detection", "conversation"]:  # Show Stop button if active
            if st.button("Stop Ringbell System", key="stop_detection_button"):
                st.session_state.current_phase = "setup"
                st.session_state.agent_running = False
                st.session_state.listening_for_user = False
                st.session_state.conversation_started = False  # Reset flag
                st.session_state.camera_active = False
                st.session_state.chat_scroll_to_bottom = False  # Reset scroll
                st.session_state.user_text_input_key = 0  # Reset text input key
                # Release camera explicitly
                if st.session_state.cap is not None and st.session_state.cap.isOpened():
                    st.session_state.cap.release()
                    st.session_state.cap = None
                st.info("Ringbell system stopped.")
                st.rerun()  # Trigger rerun to update UI to setup phase

# --- Main Application Logic based on current_phase ---

if st.session_state.current_phase == "detection":
    # Use st.status to provide persistent feedback during detection
    with st.status("Initializing Camera and Looking for Visitors...", expanded=True,
                   state="running") as detection_status:
        video_output = video_placeholder.empty()  # Placeholder for live video feed
        current_status_text = status_placeholder.empty()  # Placeholder for status messages

        # Check if camera is active and opened
        if not st.session_state.camera_active or st.session_state.cap is None or not st.session_state.cap.isOpened():
            detection_status.update(label="Camera Not Active or Failed to Open", state="error", expanded=True)
            current_status_text.error(
                "Camera is not active or failed to open. Check permissions and ensure no other app is using it.")
            st.session_state.current_phase = "setup"
            # Ensure camera is released if it was partially opened
            if st.session_state.cap is not None and st.session_state.cap.isOpened():
                st.session_state.cap.release()
                st.session_state.cap = None
            st.session_state.camera_active = False
            st.stop()  # Halt current execution

        # Get a single frame from the camera and perform detection
        frame_data, detected_name, status_message = get_camera_feed_and_detect(
            st.session_state.face_app, st.session_state.collection, st.session_state.cap
        )

        if frame_data:
            video_output.image(frame_data, channels="BGR", use_container_width=True)  # Changed to use_container_width
            current_status_text.text(status_message)
            detection_status.update(label=f"Camera Active: {status_message}", state="running", expanded=True)

            if detected_name:
                st.session_state.visitor_name = detected_name
                st.session_state.current_phase = "conversation"
                st.session_state.agent_running = True
                st.session_state.conversation_started = False  # Reset for fresh conversation start
                st.session_state.chat_scroll_to_bottom = True  # Prepare to scroll chat
                detection_status.update(label=f"Visitor identified: {detected_name}. Starting VBot conversation.",
                                        state="complete", expanded=False)
                # Release camera as detection phase is ending
                if st.session_state.cap is not None and st.session_state.cap.isOpened():
                    st.session_state.cap.release()
                    st.session_state.cap = None
                st.session_state.camera_active = False
                st.rerun()  # Trigger rerun to switch to conversation phase
            else:
                time.sleep(0.01)  # Small delay to prevent burning CPU
                st.rerun()  # Force rerun to get the next frame and continue detection
        else:
            # Handle cases where frame_data is None (e.g., webcam error or encoding error)
            detection_status.update(label=f"Detection Error: {status_message}", state="error", expanded=True)
            current_status_text.error(status_message)
            st.session_state.current_phase = "setup"
            # Ensure camera is released on error
            if st.session_state.cap is not None and st.session_state.cap.isOpened():
                st.session_state.cap.release()
                st.session_state.cap = None
            st.session_state.camera_active = False
            st.stop()


elif st.session_state.current_phase == "conversation":
    # --- Step 1: Initialize graph and trigger first AI response for the FIRST time ---
    if not st.session_state.conversation_started:
        st.session_state.chat_messages = []  # Clear chat for new conversation
        st.session_state.user_text_input_key = 0  # Reset text input key for new conversation

        initial_human_message = HumanMessage(content=f"My name is {st.session_state.visitor_name}")
        st.session_state.chat_messages.append(initial_human_message)  # Add initial message to display

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
        st.session_state.conversation_started = True  # Mark conversation as started
        st.session_state.chat_scroll_to_bottom = True  # Prepare to scroll chat

        # IMPORTANT: Invoke graph immediately after initialization for first AI response
        try:
            with status_placeholder.container():
                with st.spinner("VBot thinking..."):
                    # Use the just-initialized state for the first invocation
                    new_graph_state = graph.invoke(st.session_state.vbot_graph_state)

            st.session_state.vbot_graph_state = new_graph_state
            # Add new messages generated by the graph invocation to chat history
            # Only add messages that are not already in chat_messages to avoid duplicates
            new_messages_from_graph = [
                msg for msg in new_graph_state["messages"]
                if msg not in st.session_state.chat_messages
            ]
            st.session_state.chat_messages.extend(new_messages_from_graph)

            # Explicitly display new AI messages to chat_container and speak them
            # This is done here to ensure immediate display before rerun.
            with chat_container:
                for msg in new_messages_from_graph:
                    if isinstance(msg, AIMessage):
                        with st.chat_message("assistant"):
                            st.write(
                                msg.content if msg.content else "[AI message - no content]")  # Use fallback for display too
                        if msg.content:  # Speak only if content is not empty
                            speak_text(msg.content)
                    # Tool messages are internal processing, not conversational output for the user
                    # These are usually not displayed to the user in the main chat flow
                    # elif isinstance(msg, ToolMessage):
                    #     with st.chat_message("tool"):
                    #         st.write(f"Tool Output (Internal):") # Label as internal
                    #         try:
                    #             st.json(json.loads(msg.content))
                    #         except json.JSONDecodeError:
                    #             st.write(msg.content)

            st.rerun()  # Trigger rerun to display the initial AI response and trigger next loop
        except Exception as e:
            st.error(f"Error during initial agent invocation: {e}")
            st.session_state.agent_running = False
            st.session_state.current_phase = "setup"
            st.stop()

    # --- Step 2: Handle ongoing conversation based on state ---
    # Retrieve current_graph_state safely for all subsequent logic in this block
    current_graph_state = st.session_state.vbot_graph_state
    if current_graph_state is None:  # Defensive check, should ideally not be hit with conversation_started flag
        st.error("VBot graph state is unexpectedly None during ongoing conversation. Re-starting.")
        st.session_state.current_phase = "setup"
        st.session_state.agent_running = False
        st.session_state.conversation_started = False
        st.stop()

    # If conversation is finished, display info and transition back
    if current_graph_state.get("finished", False):
        st.info("VBot conversation finished. Returning to detection mode in 5 seconds...")
        st.session_state.agent_running = False
        st.session_state.current_phase = "detection"
        st.session_state.conversation_started = False  # Reset flag for next conversation
        st.session_state.chat_scroll_to_bottom = True  # Prepare to scroll chat for final message
        # Ensure the final message is displayed before transitioning
        if st.session_state.vbot_graph_state and st.session_state.vbot_graph_state["messages"]:
            final_msg = st.session_state.vbot_graph_state["messages"][-1]
            # Only display AIMessage content, not internal ToolMessages
            if isinstance(final_msg, AIMessage) and final_msg not in st.session_state.chat_messages:
                st.session_state.chat_messages.append(final_msg)
        time.sleep(5)  # Give user a moment to read the final message
        st.rerun()  # Trigger rerun to switch back to detection phase

    # Continue if agent is running and conversation is NOT finished
    elif st.session_state.agent_running:
        # Check if it's currently a human turn (graph indicates 'human' as next_node)
        # and we are not currently processing user input (which happens after 'Send' click)
        current_graph_state = st.session_state.vbot_graph_state
        next_node_options = graph.get_next_steps(current_graph_state)

        if "human" in next_node_options:
            # If graph expects human input, display the text input and send button
            # We explicitly *don't* call graph.invoke() here.
            pass  # UI elements are outside this block, just above

        # If it's not a human turn, and we are not listening for user input, then auto-progress
        elif not st.session_state.listening_for_user:  # This condition is now about auto-progression
            # This block automatically invokes the graph if the last message in chat was from the AI
            # or a Tool, meaning VBot needs to continue its internal processing (e.g., tool outputs,
            # or another AI turn). If the last message is a HumanMessage (and not processed yet), it waits.
            if st.session_state.chat_messages and \
                    (isinstance(st.session_state.chat_messages[-1], AIMessage) or \
                     isinstance(st.session_state.chat_messages[-1], ToolMessage)):

                # If next_node is NOT 'human' (meaning it's 'tools' or 'chatbot' for internal processing),
                # then automatically invoke the graph.
                if "human" not in next_node_options:  # Ensure it's not a human turn
                    try:
                        with status_placeholder.container():
                            with st.spinner("VBot thinking..."):
                                new_graph_state = graph.invoke(current_graph_state)

                        st.session_state.vbot_graph_state = new_graph_state
                        new_messages_from_graph = [
                            msg for msg in new_graph_state["messages"]
                            if msg not in st.session_state.chat_messages
                        ]
                        st.session_state.chat_messages.extend(new_messages_from_graph)
                        st.session_state.chat_scroll_to_bottom = True  # Prepare to scroll chat

                        # Explicitly display new AI messages to chat_container
                        with chat_container:
                            for msg in new_messages_from_graph:
                                if isinstance(msg, AIMessage):
                                    if msg.content:
                                        with st.chat_message("assistant"):
                                            st.write(msg.content)
                                        speak_text(msg.content)
                                # Tool messages are internal processing, not conversational output for the user
                                # elif isinstance(msg, ToolMessage):
                                #     with st.chat_message("tool"):
                                #         st.write(f"Tool Output (Internal):")
                                #         try:
                                #             st.json(json.loads(msg.content))
                                #         except json.JSONDecodeError:
                                #             st.write(msg.content)

                        st.rerun()  # Trigger rerun to display new messages and continue auto-flow
                    except Exception as e:
                        st.error(f"Error during agent auto-processing: {e}")
                        st.session_state.agent_running = False
                        st.session_state.current_phase = "setup"
                        st.stop()
            # If the last message was a HumanMessage, it means we are awaiting user input.
            # No graph invoke or rerun needed here. The text input field is shown by the UI block above.
            else:
                pass






