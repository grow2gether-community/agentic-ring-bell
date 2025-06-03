from vbot_graph import graph
from langchain_core.messages import HumanMessage
from time import sleep
from face_detection import detect_face_and_greet
from vbot_graph import speak_text

def run_agent_for_visitor(name: str, owner_status):
    state = {
        "messages": [HumanMessage(content=f"My name is {name}")],
        "authenticated": False,
        "finished": False,
        "owner_status": owner_status,
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

# === IMAGE-RECOGNITION TRIGGER ===
def start_ringbell():
    owner_status = get_owner_status()
    while True:
        detected_name = detect_face_and_greet()
        if detected_name == "Unknown":
            speak_text("Visitor not recognized. Ending interaction.")
            print("Visitor not recognized. Ending interaction.\n")
            continue
        print(f"Detected: {detected_name}")
        run_agent_for_visitor(detected_name, owner_status)
        sleep(5)

if __name__ == "__main__":
    start_ringbell()
