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

# @tool
# def update_frequency(user_name: str) -> dict:
#     """Increments frequency count for a verified user."""
#     user = USER_DB.get(user_name)
#     if user:
#         user["frequency"] += 1
#         return {"frequency_updated": True}
#     return {"frequency_updated": False}

# === Tools ===

@tool
def verify_user(user_name: str) -> dict:
    """
    Verifies if the user exists in the system and provides context like frequency of visits.
    """
    user = USER_DB.get(user_name)
    if not user:
        return {
            "user_found": False,
            "frequency": 0,
            "finished": True
        }

    return {
        "user_found": True,
        "frequency": USER_DB[user_name][1],
        "finished": False
    }

@tool
def verify_otp(user_name: str, otp: str) -> dict:
    """
    Verifies the OTP for a user. Returns only whether it was correct.
    """
    user = USER_DB.get(user_name)
    if not user:
        return {"otp_correct": False, "user_found": False, "finished": True}

    correct_otp, _ = user
    if otp != correct_otp:
        return {"otp_correct": False, "user_found": True, "finished": True}

    # Don't update frequency here — let chatbot decide and call a tool if needed
    return {"otp_correct": True, "user_found": True, "finished": True}

# # === Graph Setup ===
tools = [verify_user, verify_otp]
tool_node = ToolNode(tools)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm.bind_tools(tools)


def generate_prompt(owner_status: str) -> tuple:
    return (
        "system",
        f"""You are VBot, a smart and polite voice assistant at the front door.
        The owner is currently marked as '{owner_status}'.

        You will receive:
        - A greeting like "My name is X"
        - Then tool outputs (e.g., user_found, frequency, authenticated, otp_correct, owner_status, finished)

        Based on that:
        1. If `user_found` is False:
        → Politely say the user is not recognized and cannot proceed.
        2. If `user_found` is True:
        → Greet them based on their `frequency`. If high, make it warmer.
        3. If `authenticated` is False:
        → Prompt for the OTP briefly and clearly.
        4. If `authenticated` is True:
        → Decide the final response based on `owner_status`:
            - If 'home' → Welcome them in.
            - If 'away' and frequency > 10 → Say they're trusted and allowed in.
            - If 'away' and frequency <= 10 → Say access is denied.
            - If 'out_of_place' → Say owner is unavailable and entry is denied.
        5. Keep your tone friendly, professional, and empathetic.

        Your final response must always be **a single, user-friendly sentence**.
        """
            )

def chatbot_with_tools(state: SessionState) -> SessionState:
    VBOT_SYSINT = generate_prompt(state.get("owner_status", "home"))

    if state["messages"]:
        new_output = llm_with_tools.invoke([VBOT_SYSINT] + state["messages"])


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
