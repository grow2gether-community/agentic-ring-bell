import os
from typing import Annotated, Literal
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import json
import tempfile
import speech_recognition as sr
from gtts import gTTS
import pygame
import time
import uuid

load_dotenv()


# --- speak_text and listen_for_audio_once are utility functions for app.py ---
def speak_text(text: str):
    """Convert text to speech and play using pygame."""
    if not text:
        print("Warning: speak_text called with empty text. Skipping audio.")
        return

    temp_file = os.path.join(tempfile.gettempdir(), f"vbot_audio_{os.getpid()}_{time.time_ns()}.mp3")

    try:
        tts = gTTS(text, lang='en')
        tts.save(temp_file)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            continue
    except pygame.error as e:
        print(f"Pygame audio error: {e}. Skipping audio playback.")
    except Exception as e:
        print(f"Error during speak_text: {e}")
    finally:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.quit()
        if os.path.exists(temp_file):
            os.remove(temp_file)


def listen_for_audio_once() -> str:
    """Listen for a short duration and transcribe audio."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Recording audio... (Check terminal for status)")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout.")
            return ""
        except Exception as e:
            print(f"Microphone listening error: {e}")
            return ""

    print("Processing audio...")
    try:
        result = recognizer.recognize_google(audio)
        return result.strip()
    except sr.UnknownValueError:
        print("❌ Could not understand the audio. Please try again.")
    except sr.RequestError as e:
        print(f"❌ Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"An unexpected error occurred during transcription: {e}")
    return ""


api_key = os.getenv("GOOGLE_API_KEY")


# === Session State ===
class SessionState(TypedDict):
    messages: Annotated[list, add_messages]
    authenticated: bool
    finished: bool
    owner_status: Literal["home", "away", "out_of_place"]
    delivery_expected: bool
    frequency_updated: bool
    waiting_for_human: bool 


# === Simulated User DB ===
USER_DB = {
    "subbu": ["1234", 9],
    "ram": ["1234", 12],
    "ravi": ["1234", 3],
    "delivery_agent": ["", 0],
}


# === Tools ===

@tool
def deliver_message(username: str) -> dict:
    """Looks for any expected deliveries, if expected delivery is found,
    it delivers a message to the user, when the user is 'delivery_agent'"""
    return {"user_found": True, "finished": True}


# === Graph Setup ===
tools = [deliver_message]
tool_node = ToolNode(tools)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
llm_with_tools = llm.bind_tools(tools)


def generate_prompt(owner_status: str, delivery_expected: bool) -> tuple:
    return (
        "system",
        f"""You are VBot, a smart and polite voice assistant at the front door.
        The owner is currently marked as '{owner_status}', and delivery is expected: '{delivery_expected}'.

        You will receive a greeting like "My name is X". Based on the name:

        1. If name is 'delivery_agent':
        → If delivery_expected is True: Deliver an interactive message (e.g., "We've been waiting for this delivery!")
        → If delivery_expected is False: Show curiosity about the delivery (e.g., "What's in the package?")
        → Always be friendly and engaging - share fun facts, ask about their day, etc.

        2. If name is 'unknown':
        → Politely say you couldn't recognize them and cannot proceed.
        → Keep the tone professional but firm.

        3. For any other name (recognized person):
        → Greet them warmly by name
        → Based on owner_status:
            - If 'home': Welcome them in
            - If 'away': Say the owner is not home and entry is denied
            - If 'out_of_place': Say the owner is unavailable and entry is denied
        → Keep your tone friendly and professional

        Always maintain a polite and professional demeanor while being clear about access decisions.
        """
    )

def chatbot_with_tools(state: SessionState) -> SessionState:
    VBOT_SYSINT = generate_prompt(state.get("owner_status", "home"), state.get("delivery_expected", False))

    print("\nDEBUG: Current messages in state (chatbot_with_tools entry):")
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"AI: {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"Tool calls: {msg.tool_calls}")
        elif isinstance(msg, ToolMessage):
            print(f"Tool: {msg.content}")

    new_output = llm_with_tools.invoke([VBOT_SYSINT] + state["messages"])

    print("\nDEBUG: New output from LLM (chatbot_with_tools):")
    if isinstance(new_output, AIMessage):
        print(f"Content: {new_output.content}")
        if hasattr(new_output, 'tool_calls') and new_output.tool_calls:
            print(f"Tool calls: {new_output.tool_calls}")

    if isinstance(new_output, AIMessage) and not new_output.content:
        if hasattr(new_output, 'tool_calls') and new_output.tool_calls:
            pass
        else:
            new_output.content = "I apologize, I seem to have lost my thought. Could you please repeat that or tell me more?"
            print("DEBUG: LLM produced empty content without tool calls. Applying fallback.")

    new_state = {**state, "messages": state["messages"] + [new_output]}

    # Process tool messages for state updates
    for msg in [new_output]:
        if isinstance(msg, ToolMessage):
            try:
                tool_output = json.loads(msg.content)
                if "finished" in tool_output:
                    new_state["finished"] = tool_output["finished"]
            except Exception as e:
                print(f"Error processing tool message: {e}")

    # IMPORTANT: Set finished=True after responding, unless a tool call indicates otherwise
    if not new_state.get("finished", False):
        new_state["finished"] = True
        print("DEBUG: Setting finished=True after chatbot response")

    print(f"\nDEBUG: Final state (chatbot_with_tools exit) - finished: {new_state.get('finished')}")
    return new_state


def route_from_chatbot(state: SessionState) -> str:
    print(f"\nDEBUG: route_from_chatbot called. Current state finished: {state.get('finished')}")

    if state.get("finished", False):
        print("DEBUG: Graph state finished is True, routing to END.")
        return END
    
    last_msg = state["messages"][-1]
    
    # If the last message has tool calls, route to tools
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print("DEBUG: Last message has tool calls, routing to tools.")
        return "tools"
    
    # Default routing: if not finished and no tools, continue through the chatbot node
    print("DEBUG: Default routing to chatbot.")
    return "chatbot"


# === Build Graph ===
graph_builder = StateGraph(SessionState)
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", route_from_chatbot)
graph_builder.add_edge("tools", "chatbot")
# Add direct edge to END when finished
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()