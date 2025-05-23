# from langchain_core.messages import HumanMessage, AIMessage
# from time import sleep
# import os
# from gtts import gTTS
# import speech_recognition as sr
# from vbot_graph import graph  # <- this is your LangGraph graph instance

# # === SPEECH UTILS ===

# def speak(text: str):
#     tts = gTTS(text)
#     filename = "/tmp/voice.mp3"
#     tts.save(filename)
#     os.system(f"start {filename}" if os.name == 'nt' else f"afplay {filename}")

# def listen() -> str:
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening...")
#         audio = recognizer.listen(source)
#     try:
#         result = recognizer.recognize_google(audio)
#         print(f"You said: {result}")
#         return result
#     except sr.UnknownValueError:
#         return "Sorry, I didn't catch that."

# # === AGENT RUNNER ===

# def run_agent_for_visitor(name: str, owner_status="home"):
#     state = {
#         "messages": [HumanMessage(content=f"My name is {name}")],
#         "authenticated": False,
#         "finished": False,
#         "owner_status": owner_status
#     }

#     while not state.get("finished"):
#         state = graph.invoke(state)

#         last_msg = state["messages"][-1]
#         if isinstance(last_msg, AIMessage):
#             print("Model:", last_msg.content)
#             speak(last_msg.content)

#         if not state.get("finished"):
#             user_response = listen()
#             state["messages"].append(HumanMessage(content=user_response))

# # === IMAGE-RECOGNITION TRIGGER ===
# def start_ringbell():
#     while True:
#         detected_name = check_camera_for_person()  # <-- Your team plugs this in
#         if detected_name:
#             print(f"Detected: {detected_name}")
#             run_agent_for_visitor(detected_name)
#             sleep(5)  # Prevent immediate re-trigger

# # === Replace with your actual image recognition hook ===
# def check_camera_for_person():
#     # Simulated test for now
#     input("Press Enter to simulate 'Subbu' detected...")
#     return "Subbu"

# # === MAIN ===
# if __name__ == "__main__":
#     start_ringbell()

import os
import tempfile
from time import sleep
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from langchain_core.messages import HumanMessage, AIMessage
from vbot_graph import graph

# === SPEECH UTILITIES ===

# def speak_text(text: str):
#     """Convert given text to speech, play it, then delete it from disk."""
#     filename = "temp_speech.mp3"
#     tts = gTTS(text)
#     tts.save(filename)

#     try:
#         os.system(f"start {filename}" if os.name == "nt" else f"afplay {filename}")
#     finally:
#         if os.path.exists(filename):
#             os.remove(filename)


#caption = "Can you please tell your name?"
# def speak_text(text: str):
#     """Convert given text to speech, play it, then delete it from disk."""
# #Create a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
#         tts = gTTS(text, lang='en')
#         tts.save(fp.name)
#         temp_file = fp.name

#     # Play after generation
#     playsound(temp_file)

#     # Optional: cleanup
#     os.remove(temp_file)

# def listen_prompted() -> str:
#     """Prompt user to press Enter to start and stop recording, then transcribe."""
#     recognizer = sr.Recognizer()
#     mic = sr.Microphone()

#     input("🎤 Press Enter to start recording your response...")
#     print("Recording... Press Enter again when done.")
    
#     with mic as source:
#         recognizer.adjust_for_ambient_noise(source)
#         audio = recognizer.listen(source)
    
#     input("🛑 Recording stopped. Press Enter to process.")

#     try:
#         result = recognizer.recognize_google(audio)
#         print(f"You said: {result}")
#         return result.strip()
#     except sr.UnknownValueError:
#         print("❌ Could not understand the audio.")
#         return ""

# === AGENT LOOP ===
from vbot_graph import graph
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
