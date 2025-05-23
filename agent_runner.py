from vbot_graph import graph
from langchain_core.messages import HumanMessage
from time import sleep

def run_agent_for_visitor(name: str, owner_status="home"):
    state = {
        "messages": [HumanMessage(content=f"My name is {name}")],
        "authenticated": False,
        "finished": False,
        "owner_status": owner_status
    }

    while not state.get("finished"):
        state = graph.invoke(state)

# === IMAGE-RECOGNITION TRIGGER ===

def check_camera_for_person():
    input("📸 Press Enter to simulate 'Subbu' detected...")
    return "Subbu"

def start_ringbell():
    while True:
        detected_name = check_camera_for_person()
        if detected_name:
            print(f"Detected: {detected_name}")
            run_agent_for_visitor(detected_name)
            sleep(5)

if __name__ == "__main__":
    start_ringbell()
