from dotenv import load_dotenv
import os
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv('OPEN_API_KEY')
from IPython.display import Image, display
from langchain_core.tools import tool
import pyttsx3
import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import uuid
import chromadb
from chromadb.config import Settings



embedding_size = 512
#index = faiss.IndexFlatL2(embedding_size)
known_faces = {}  #


### speaking 

def speak_text(text):
    '''it give the llm ability to speak '''
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 190)      # Adjust speech rate
        engine.setProperty('volume', 0.9)    # Set volume (0.0 to 1.0)

        # Optional: Choose a voice (0: male, 1: female, etc.)
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)

        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("[TTS Error] pyttsx3 failed:", e)


###used for detecting distance 





# === Configuration ===
KNOWN_FACE_HEIGHT_CM = 20.0     # Average face height in cm
FOCAL_LENGTH = 500              # Estimated focal length
TRIGGER_DISTANCE_CM = 100       # Action threshold (1 meter)
MIN_FACE_HEIGHT_PX = 40         # Ignore too-small faces

# === Initialize Face Detection ===
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# === Estimate distance ===
def estimate_distance(bbox, known_height=KNOWN_FACE_HEIGHT_CM, focal_length=FOCAL_LENGTH):
    x1, y1, x2, y2 = bbox
    height_px = abs(y2 - y1)
    if height_px < MIN_FACE_HEIGHT_PX:
        return float('inf')
    return (known_height * focal_length) / height_px

# === Triggered action ===

    
    
 

# === Start webcam ===
def look_person():
    """It detects person if the person is less than 100cm from the camera of else it loops until the person is 100 less than 100 cm
    and it compares the detected face with embeddings in chromadb if matches the face it returns name and distance to between them"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        exit()

    print("📷 Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_app.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int)
            distance_cm = estimate_distance(bbox)

            if distance_cm == float('inf') or distance_cm > 500:
                continue

            # Draw bounding box + distance
            '''label = f"{distance_cm:.1f} cm"
            color = (0, 255, 0) if distance_cm < TRIGGER_DISTANCE_CM else (255, 0, 0)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Distance Estimation", frame)
                        '''

            if distance_cm < TRIGGER_DISTANCE_CM:
                cap.release()
                cv2.destroyAllWindows()
                #trigger_function(face)
                emb = face.embedding

                name, dist = recognize_face(emb)
                if name:
        #print(f"Recognized {name} with distance {dist:.2f}")
                    return f" you are working as smart security ring camera ai to a house Greet {name} who has about to enter house"
                else:
                    return "you are working as smart security ring camera ai to a house you just seen a unknown person who is not in your database greet him in human way by saying your at 215 peck avenue how can i help you today !"
        
        # Show the webcam output
        

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # === Cleanup ===
    cap.release()
    cv2.destroyAllWindows()




client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="face_embeddings", metadata={"hnsw:space": "cosine"})


def normalize(vec):
    return vec / np.linalg.norm(vec)

def add_known_face(name, image_path):
    img = cv2.imread(image_path)
    faces = face_app.get(img)
    if not faces:
        print(f"No face found in {image_path}")
        return

    emb = normalize(faces[0].embedding).tolist()
    doc_id = str(uuid.uuid4())  # Unique ID for this face

    collection.add(
        ids=[doc_id],
        embeddings=[emb],
        metadatas=[{"name": name}],
        documents=[image_path]
    )
    print(f"Added {name}'s face to ChromaDB.")

## recognizing known faces in chroma db
def recognize_face(embedding, threshold=0.49):
    embedding = normalize(embedding).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=1)

    if results["distances"][0][0] > threshold:
        return None, results["distances"][0][0]

    matched_name = results["metadatas"][0][0]["name"]
    return matched_name, results["distances"][0][0]

## upload images of persons
add_known_face("Raghav", "./image/rag1.jpg")
add_known_face("Raghav", "./image/rag2.jpg")
add_known_face("Raghav", "./image/rag3.jpg")
add_known_face("Raghav", "./image/rag4.jpg")
add_known_face("Vamsi", "./image/vam.jpg")


#####  creating agent

from langchain.chat_models import init_chat_model

from langgraph.prebuilt import create_react_agent



model = init_chat_model("openai:gpt-4o-mini", temperature=0)
tools = [speak_text,look_person]
agent = create_react_agent(
    # disable parallel tool calls
    model=model.bind_tools(tools, parallel_tool_calls=False),
    tools=tools
)

agent.invoke(
    {"messages": [{"role": "user", "content": "look any person in the front camera if the person is detected say hello"}]}
)


