from vbot_graph import graph
from langchain_core.messages import HumanMessage
from time import sleep
from face_detection import detect_face_and_greet

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
def start_ringbell():
    while True:
        detected_name = detect_face_and_greet()
        if detected_name:
            print(f"Detected: {detected_name}")
            run_agent_for_visitor(detected_name)
            sleep(5)

if __name__ == "__main__":
    start_ringbell()
