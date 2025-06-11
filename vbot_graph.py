# import os
# from typing import Annotated, Literal
# from dotenv import load_dotenv
# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# from langchain_core.tools import tool
# import json
# import tempfile
# import speech_recognition as sr
# from gtts import gTTS
# import pygame
#
# load_dotenv()
# def speak_text(text: str):
#     """Convert text to speech and play using pygame."""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
#         tts = gTTS(text)
#         tts.save(fp.name)
#         temp_file = fp.name
#
#     pygame.init()
#     pygame.mixer.init()
#     pygame.mixer.music.load(temp_file)
#     pygame.mixer.music.play()
#
#     while pygame.mixer.music.get_busy():
#         continue
#
#     pygame.mixer.music.stop()
#     pygame.quit()
#     os.remove(temp_file)
#
# def listen_prompted() -> str:
#     """Prompt user to press Enter to start and stop recording, then transcribe."""
#     recognizer = sr.Recognizer()
#     mic = sr.Microphone()
#
#     input("🎤 Press Enter to start recording your response...")
#     print("Recording... Press Enter again when done.")
#
#     with mic as source:
#         recognizer.adjust_for_ambient_noise(source)
#         audio = recognizer.listen(source)
#
#     input("🛑 Recording stopped. Press Enter to process.")
#
#     try:
#         result = recognizer.recognize_google(audio)
#         print(f"You said: {result}")
#         return result.strip()
#     except sr.UnknownValueError:
#         print("❌ Could not understand the audio.")
#         return ""
#
# # #save the api key in the environment variable
# api_key = os.getenv("GOOGLE_API_KEY")
#
# # === Session State ===
# class SessionState(TypedDict):
#     messages: Annotated[list, add_messages]
#     authenticated: bool
#     finished: bool
#     owner_status: Literal["home", "away", "out_of_place"]
#     delivery_expected: bool
#     frequency_updated: bool
#
# # === Simulated User DB ===
# USER_DB = {
#     "subbu": ["1234", 9],
#     "ram": ["1234", 12],
#     "ravi": ["1234", 3],
# }
#
# # @tool
# # def update_frequency(user_name: str) -> dict:
# #     """Increments frequency count for a verified user."""
# #     user = USER_DB.get(user_name)
# #     if user:
# #         user["frequency"] += 1
# #         return {"frequency_updated": True}
# #     return {"frequency_updated": False}
#
# # === Tools ===
#
# @tool
# def verify_user(user_name: str) -> dict:
#     """
#     Verifies if the user exists in the system and provides context like frequency of visits.
#     """
#
#     user = USER_DB.get(user_name)
#     if not user:
#         return {
#             "user_found": False,
#             "frequency": 0,
#             "finished": True
#         }
#
#     return {
#         "user_found": True,
#         "frequency": USER_DB[user_name][1],
#         "finished": False
#     }
#
# @tool
# def verify_otp(user_name: str, otp: str) -> dict:
#     """
#     Verifies the OTP for a user. Returns only whether it was correct.
#     """
#     user = USER_DB.get(user_name)
#     if not user:
#         return {"otp_correct": False, "user_found": False, "finished": True}
#
#     correct_otp, _ = user
#     if otp != correct_otp:
#         return {"otp_correct": False, "user_found": True, "finished": True}
#
#     # Don't update frequency here — let chatbot decide and call a tool if needed
#     return {"otp_correct": True, "user_found": True, "finished": True}
#
# @tool
# def deliver_message(username: str) -> dict:
#     """Looks for any expected deliveries, if expected delivery is found,
#     it delivers a message to the user, when the user is 'delivery'"""
#     return {"user_found": True, "finished": True}
#
# # # === Graph Setup ===
# tools = [verify_user, verify_otp, deliver_message]
# tool_node = ToolNode(tools)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
# llm_with_tools = llm.bind_tools(tools)
#
#
# def generate_prompt(owner_status: str, delivery_expected: bool) -> tuple:
#     return (
#         "system",
#         f"""You are VBot, a smart and polite voice assistant at the front door.
#         The owner is currently marked as '{owner_status}', and delivery is expected: '{delivery_expected}'.
#
#         You will receive:
#         - A greeting like "My name is X"
#         - Then tool outputs (e.g., user_found, frequency, authenticated, otp_correct, owner_status, finished)
#
#         If user_name is 'delivery_agent':
#         → Deliver a message to the user, by considering the 'delivery_expected' flag.
#         → If delivery_expected is True, deliver a message to the user thats interactive(like you have been waiting for it, thanking the deliver person etc.)
#         → If delivery_expected is False, deliver a message to the user thats shows curiosity about the delivery(like, whats in there, I am excited about it etc,)
#         → You should always tell some fun facts, jokes, or interesting information; ask them politely about their day etc.
#
#         If user_name is not 'delivery_agent':
#         1. If `user_found` is False:
#         → Politely say the user is not recognized and cannot proceed.
#         2. If `user_found` is True:
#         → Greet them based on their `frequency`. If high, make it warmer.
#         3. If `authenticated` is False:
#         → Prompt for the OTP briefly and clearly.
#         4. If `authenticated` is True:
#         → Decide the final response based on `owner_status`:
#             - If 'home' → Welcome them in.
#             - If 'away' and frequency > 10 → Say they're trusted and allowed in.
#             - If 'away' and frequency <= 10 → Say access is denied.
#             - If 'out_of_place' → Say owner is unavailable and entry is denied.
#         5. Keep your tone friendly, professional, and empathetic.
#         """
#             )
#
# def chatbot_with_tools(state: SessionState) -> SessionState:
#     VBOT_SYSINT = generate_prompt(state.get("owner_status", "home"), state.get("delivery_expected", False))
#
#     if state["messages"]:
#         new_output = llm_with_tools.invoke([VBOT_SYSINT] + state["messages"])
#
#
#     for msg in state["messages"]:
#         if isinstance(msg, ToolMessage):
#             try:
#                 parsed = json.loads(msg.content)
#                 if "authenticated" in parsed:
#                     state["authenticated"] = parsed["authenticated"]
#                 if "finished" in parsed:
#                     state["finished"] = parsed["finished"]
#             except:
#                 pass
#
#     if state.get("finished"):
#         print("Model:", new_output.content)
#         speak_text(new_output.content)
#     return {**state, "messages": state["messages"] + [new_output]}
#
# def human_node(state: SessionState) -> SessionState:
#     last_msg = state["messages"][-1]
#     print("Model:", last_msg.content)
#     speak_text(last_msg.content)
#
#     user_input = listen_prompted()
#     print("User:", user_input)
#     return state | {"messages": [HumanMessage(content=user_input)]}
#
#
# def route_from_chatbot(state: SessionState) -> str:
#     if state.get("finished"):
#         return END
#     last_msg = state["messages"][-1]
#     if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
#         return "tools"
#     return "human"
#
# # === Build Graph ===
# graph_builder = StateGraph(SessionState)
# graph_builder.add_node("chatbot", chatbot_with_tools)
# graph_builder.add_node("tools", tool_node)
# graph_builder.add_node("human", human_node)
# graph_builder.add_conditional_edges("chatbot", route_from_chatbot)
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge("human", "chatbot")
# graph_builder.add_edge(START, "chatbot")
# graph = graph_builder.compile()

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


def listen_for_audio_once() -> str:  # This is the function called by app.py for speech input
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
        return ""
    except sr.RequestError as e:
        print(f"❌ Could not request results from Google Speech Recognition service; {e}")
        return ""
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


# === Simulated User DB ===
USER_DB = {
    "subbu": ["1234", 9],
    "ram": ["1234", 12],
    "ravi": ["1234", 3],
    "delivery_agent": ["", 0],
}


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

    return {"otp_correct": True, "user_found": True, "finished": False}


@tool
def deliver_message(username: str) -> dict:
    """Looks for any expected deliveries, if expected delivery is found,
    it delivers a message to the user, when the user is 'delivery'"""
    return {"user_found": True, "finished": True}


# === Graph Setup ===
tools = [verify_user, verify_otp, deliver_message]
tool_node = ToolNode(tools)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
llm_with_tools = llm.bind_tools(tools)


def generate_prompt(owner_status: str, delivery_expected: bool) -> tuple:
    return (
        "system",
        f"""You are VBot, a smart and polite voice assistant at the front door.
        The owner is currently marked as '{owner_status}', and delivery is expected: '{delivery_expected}'.
        You MUST always provide a textual response unless you are making a tool call. If you have no other relevant response, politely acknowledge and ask for clarification, or use a general greeting.

        You will receive:
        - A greeting like "My name is X"
        - Then tool outputs (e.g., user_found, frequency, authenticated, otp_correct, owner_status, finished)

        If user_name is 'delivery_agent':
        → Use the 'deliver_message' tool to mark completion.
        → If delivery_expected is True, deliver an interactive message showing you've been waiting, thanking them, etc.
        → If delivery_expected is False, deliver a curious message about the delivery (e.g., "What's in there? I'm excited!")
        → Always include a fun fact, joke, or interesting information; politely ask about their day.

        If user_name is not 'delivery_agent':
        1. If `user_found` is False:
        → Politely state the user is not recognized and cannot proceed. Set 'finished' to True.
        2. If `user_found` is True:
        → Greet them based on their `frequency`. If high (e.g., >10), make it warmer.
        3. If `authenticated` is False (and `user_found` is True):
        → Prompt for the OTP briefly and clearly.
        4. If `authenticated` is True (and `user_found` is True):
        → Decide the final response based on `owner_status`:
            - If 'home' → Welcome them in. Set 'finished' to True.
            - If 'away' and frequency > 10 → Say they're trusted and allowed in. Set 'finished' to True.
            - If 'away' and frequency <= 10 → Say access is denied. Set 'finished' to True.
            - If 'out_of_place' → Say owner is unavailable and entry is denied. Set 'finished' to True.
        5. Keep your tone friendly, professional, and empathetic.
        """
    )


def chatbot_with_tools(state: SessionState) -> SessionState:
    VBOT_SYSINT = generate_prompt(state.get("owner_status", "home"), state.get("delivery_expected", False))

    new_output = llm_with_tools.invoke([VBOT_SYSINT] + state["messages"])

    # Ensure AIMessage content is never empty unless a tool call is made
    if isinstance(new_output, AIMessage) and not new_output.content:
        if hasattr(new_output, 'tool_calls') and new_output.tool_calls:
            pass  # Content is empty because AI made tool calls, which is expected.
        else:
            # AI produced empty content without tool calls, apply a robust fallback.
            new_output.content = "I apologize, I seem to have lost my thought. Could you please repeat that or tell me more?"

    # --- ADDED CLI PRINT FOR MODEL OUTPUT ---
    if isinstance(new_output, AIMessage) and new_output.content:
        print("Model:", new_output.content)

    new_state = {**state, "messages": state["messages"] + [new_output]}

    # Process tool messages for state updates (finished, authenticated)
    for msg in new_state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                parsed_content = json.loads(msg.content)
                if "authenticated" in parsed_content:
                    new_state["authenticated"] = parsed_content["authenticated"]
                if "finished" in parsed_content:
                    new_state["finished"] = parsed_content["finished"]
            except json.JSONDecodeError:
                print(f"Warning: ToolMessage content is not valid JSON: {msg.content}")
            except Exception as e:
                print(f"Error processing tool message: {e}")

    return new_state


def human_node(state: SessionState) -> SessionState:
    # This node's purpose is purely to signal that human input is expected.
    # The actual text input, displaying, and appending to state is handled by app.py.
    # We will NOT call listen_prompted or print/speak here directly.
    # --- ADDED CLI PRINT FOR HUMAN NODE SIGNAL (FOR DEBUGGING CLI FLOW) ---
    print("DEBUG: Graph routed to human_node. Awaiting input from Streamlit UI.")
    return state


def route_from_chatbot(state: SessionState) -> str:
    if state.get("finished", False):
        return END
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
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





