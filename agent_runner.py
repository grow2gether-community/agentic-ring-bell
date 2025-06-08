import time
import chromadb
from dotenv import load_dotenv
from chromadb.config import Settings
from insightface.app import FaceAnalysis
from langchain_core.messages import HumanMessage
# Import the refactored functions
from face_enrollment import run_enrollment_workflow
from face_detection import detect_visitor
from vbot_graph import graph

def run_agent_for_visitor(name: str, owner_status: str, delivery_expected: bool):
    state = {
        "messages": [HumanMessage(content=f"My name is {name}")],
        "authenticated": False,
        "finished": False,
        "owner_status": owner_status,
        "delivery_expected": delivery_expected,
        "user_name": name,
        "frequency_updated": False
    }

    while not state.get("finished"):
        state = graph.invoke(state)

def get_owner_status():
    options = {
        "1": "home",
        "2": "away",
        "3": "out_of_place"
    }
    print("Select owner status:")
    print("1. Home 🏠")
    print("2. Away 🚪")
    print("3. Out of place 🌍")

    while True:
        choice = input("Enter option (1/2/3): ").strip()
        if choice in options:
            return options[choice]
        print("Invalid option. Please select 1, 2, or 3.")

def get_delivery_expected():
    options = {
        "1": True,
        "2": False
    }
    print("Select delivery expected:")
    print("1. Yes 📦")
    print("2. No 🚫")

    while True:
        choice = input("Enter option (1/2): ").strip()
        if choice in options:
            return options[choice]
        print("Invalid option. Please select 1 or 2.")

# === Main Application Setup ===
def initialize_systems():
    """
    Initializes all necessary services like FaceAnalysis and ChromaDB.
    """
    print("Initializing systems, please wait...")
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

        print("Systems initialized.")
        return face_app, collection
    except Exception as e:
        print(f"Critical error during system initialization: {e}")
        exit()

# === Main Workflow ===
def start_ringbell(face_app, collection):
    """
    The main loop that continuously looks for visitors and triggers the agent.
    """
    owner_status = get_owner_status()
    delivery_expected = get_delivery_expected()
    while True:
        detected_name = detect_visitor(face_app, collection)

        if detected_name:
            if detected_name == "unknown":
                # speak_text("Visitor not recognized. Ending interaction.")
                print("Visitor not recognized. Ending interaction.")
            else:
                print(f"Visitor identified as: {detected_name}")
                run_agent_for_visitor(detected_name, owner_status, delivery_expected)
        else:
            # This case handles when the user quits the webcam window manually.
            print("Detection cycle ended without recognition. Press 'q' in the webcam window to exit the script.")

        print("Restarting detection cycle in 5 seconds...")
        time.sleep(5)


# === NEW: Helper function to count unique persons ===
def get_unique_person_count(collection):
    """
    Gets the number of unique persons by counting unique names in the metadata.
    """
    try:
        # Get all entries; including only metadatas is more efficient
        all_items = collection.get(include=['metadatas'])
        metadatas = all_items.get('metadatas')

        if not metadatas:
            return 0

        # Use a set to automatically handle uniqueness of names
        names = {meta['name'] for meta in metadatas if 'name' in meta}
        return len(names)
    except Exception as e:
        print(f"Could not get person count: {e}")
        return 0  # Fallback to 0 if there's an error

if __name__ == "__main__":
    face_app, collection = initialize_systems()

    num_persons = get_unique_person_count(collection)
    print(f"\nDatabase currently has {num_persons} recognized person(s).")
    enroll_choice = input("Would you like to enroll new faces now? (yes/no): ").strip().lower()

    if enroll_choice in ["yes", "y"]:
        run_enrollment_workflow(face_app, collection)
        num_persons_updated = get_unique_person_count(collection)
        print(f"\nDatabase updated. It now has {num_persons_updated} person(s).")

    input("\nPress Enter to start the live ringbell detection system...")
    start_ringbell(face_app, collection)

