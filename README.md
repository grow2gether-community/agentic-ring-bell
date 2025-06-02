# 🛎️ Agentic Ring Bell - Face Recognition Voice Assistant

A voice-interactive smart doorbell system using face recognition and an AI agent built with LangGraph. It greets known visitors, handles conversations, and supports secure access automation.

---

## 📦 Features

- 👤 Real-time face recognition using InsightFace + ChromaDB
- 🗣️ Speech-to-speech interaction using `speech_recognition` and `gTTS`
- 🧠 LangGraph agent for decision-making and conversation flow
- 🔐 Owner-aware logic (e.g., when owner is home vs away)
- 🧹 Temporary audio cleanup after speaking
- 🚫 No third-party cloud APIs required for inference

---

## ⚙️ Prerequisites

### 🧠 Python

- Python 3.10 or 3.11 (recommended)
- Virtual environment strongly recommended

### 💾 Required Packages

Install all dependencies via:

```bash
pip install -r requirements.txt

## ⚠️ Windows Users – C++ Build Tools Required

If you're on **Windows**, you need to install Microsoft C++ Build Tools before installing `insightface` or `simpleaudio`.

🛠️ These libraries depend on native C++ compilation during installation.

### 📥 Install Instructions

1. Download from: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Run the installer
3. Select the **C++ build tools** workload
4. Under optional components, check ✅ **C++ CMake tools for Windows**
5. Click Install
6. Restart your machine

> Skipping this step will cause build errors during `pip install`.


Once setup is complete, run:

> python agent_runner.py

