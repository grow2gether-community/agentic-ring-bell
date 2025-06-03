import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import json
import tempfile
from time import sleep
import speech_recognition as sr
from gtts import gTTS
import pygame
# from playsound import playsound

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


def speak_text(text: str):
    """Convert text to speech and play using pygame."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts = gTTS(text)
        tts.save(fp.name)
        temp_file = fp.name

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(temp_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

    pygame.mixer.music.stop()
    pygame.quit()
    os.remove(temp_file)

def listen_prompted() -> str:
    """Prompt user to press Enter to start and stop recording, then transcribe."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    input("🎤 Press Enter to start recording your response...")
    print("Recording... Press Enter again when done.")
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    input("🛑 Recording stopped. Press Enter to process.")

    try:
        result = recognizer.recognize_google(audio)
        print(f"You said: {result}")
        return result.strip()
    except sr.UnknownValueError:
        print("❌ Could not understand the audio.")
        return ""

#save the api key in the environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === Session State ===
class SessionState(TypedDict):
    messages: Annotated[list, add_messages]
    authenticated: bool
    finished: bool
    owner_status: Literal["home", "away", "out_of_place"]

# === Simulated User DB ===
USER_DB = {
    "Subbu": ["1234", 9],
    "Ram": ["1234", 12],
    "Ravi": ["1234", 3],
}

def update_frequency(name):
    USER_DB[name][1] += 1

# === Tools ===
@tool
def verify_user(user_name: str) -> dict:
    """
    Checks if the user is in the system.
    """
    user = USER_DB.get(user_name)
    if not user:
        return {"user_found": False, "message": f"User '{user_name}' not found.", "finished": True}
    return {"user_found": True, "message": f"Hello {user_name}, you are in the system. Please enter your OTP."}

@tool
def verify_otp(user_name: str, otp: str, owner_status: str) -> dict:
    """
    Authenticates user based on OTP and owner's current status.
    """
    user = USER_DB.get(user_name)
    if not user:
        return {"authenticated": False, "message": "User not found.", "finished": True}

    correct_otp, freq = user
    if otp != correct_otp:
        return {"authenticated": False, "message": "Incorrect OTP.", "finished": True}

    if owner_status == "home":
        update_frequency(user_name)
        return {"authenticated": True, "message": f"Welcome {user_name}, come in.", "finished": True}

    elif owner_status == "away":
        if freq > 10:
            update_frequency(user_name)
            return {"authenticated": True, "message": "Owner is away, but you're trusted. Come in.", "finished": True}
        else:
            return {"authenticated": False, "message": "Owner is away. You cannot enter.", "finished": True}

    elif owner_status == "out_of_place":
        return {"authenticated": False, "message": "Owner is out of place. Access denied.", "finished": True}

    return {"authenticated": False, "message": "Unknown error.", "finished": True}

# === Graph Setup ===
tools = [verify_user, verify_otp]
tool_node = ToolNode(tools)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# VBOT_SYSINT = (
#     "system",
#     "You are VBot. Authenticate people based on name. Ask OTP if known. "
#     "Respect owner's current status: home, away, or out_of_place. "
#     "Call verify_otp with user_name, otp, and owner_status."
# )
# WELCOME_MSG = "Welcome to VBot. May I know your name?"

def generate_prompt(owner_status: str) -> tuple:
    return (
        "system",
        f"You are VBot. The owner is currently '{owner_status}'. "
        "Authenticate people based on name. Ask OTP if the user is known. "
        "If the user is 'unknown', tell user that you can't let them in. "
        "Call verify_otp with user_name, otp, and owner_status."
    )

def chatbot_with_tools(state: SessionState) -> SessionState:
    VBOT_SYSINT = generate_prompt(state.get("owner_status", "home"))

    if state["messages"]:
        new_output = llm_with_tools.invoke([VBOT_SYSINT] + state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)

    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                parsed = json.loads(msg.content)
                if "authenticated" in parsed:
                    state["authenticated"] = parsed["authenticated"]
                if "finished" in parsed:
                    state["finished"] = parsed["finished"]
            except:
                pass

    if state.get("finished"):
        print("Model:", new_output.content)
        speak_text(new_output.content)
    return {**state, "messages": state["messages"] + [new_output]}

def human_node(state: SessionState) -> SessionState:
    last_msg = state["messages"][-1]
    print("Model:", last_msg.content)
    speak_text(last_msg.content)

    user_input = listen_prompted()
    print("User:", user_input)
    return state | {"messages": [HumanMessage(content=user_input)]}

def route_from_chatbot(state: SessionState) -> str:
    if state.get("finished"):
        return END
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tools"
    return "human"

# === Build Graph ===
graph_builder = StateGraph(SessionState)
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("human", human_node)
graph_builder.add_conditional_edges("chatbot", route_from_chatbot)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
