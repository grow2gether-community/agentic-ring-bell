# Agentic Ringbell AI System

An intelligent doorbell system that uses face recognition and AI to interact with visitors. The system can recognize known faces, handle deliveries, and provide appropriate responses based on the owner's status.

## Features

### Face Recognition
- Real-time face detection and recognition
- Face enrollment system with multiple image support
- Automatic face grouping to avoid duplicates
- Face database management (rename/delete entries)
- Confidence-based recognition with suggestions

### Owner Status Management
- Three states: home, away, out_of_place
- Delivery expectation flag
- Real-time status updates

### Visitor Interaction
- Automatic visitor identification
- Context-aware responses based on:
  - Visitor identity
  - Owner's status
  - Delivery expectations
- Natural language conversation with visitors
- Text-to-speech responses

### User Interface
- Clean, modern Streamlit interface
- Real-time webcam feed
- Live chat display
- System status monitoring
- Face management dashboard
- Scrollable enrolled people list

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentic-ring-bell.git
cd agentic-ring-bell
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Initial Setup:
   - The system will initialize the face recognition models
   - Set your owner status (home/away/out_of_place)
   - Toggle delivery expectation if needed

3. Face Enrollment:
   - Click "Enroll New Faces" in the Face Management section
   - Upload one or more images containing faces
   - The system will detect and group unique faces
   - Label each detected face
   - Skip any unrecognizable faces

4. Managing Enrolled People:
   - Click "Manage Enrolled People" to view all enrolled faces
   - Rename entries using the text input
   - Delete entries using the delete button
   - View the number of photos per person

5. Starting the System:
   - Click "Start Ringbell Detection" to begin monitoring
   - The webcam will activate and start detecting faces
   - When a visitor is detected, the system will:
     - Identify the person
     - Start a conversation based on their identity
     - Provide appropriate responses

6. Conversation Flow:
   - For recognized visitors: Greeting and access information
   - For delivery agents: Delivery handling instructions
   - For unknown visitors: Appropriate response based on owner status

7. Stopping the System:
   - Click "Stop System" to end monitoring
   - The webcam will deactivate
   - All states will be reset

## System Requirements

- Python 3.8 or higher
- Webcam
- Internet connection (for OpenAI API)
- Sufficient disk space for face database

## Dependencies

The system uses several key libraries:
- Streamlit for the web interface
- OpenCV for image processing
- InsightFace for face recognition
- LangChain for AI conversation
- ChromaDB for face database
- OpenAI for natural language processing

## Troubleshooting

1. Camera Issues:
   - Ensure no other application is using the webcam
   - Check camera permissions
   - Try restarting the application

2. Face Recognition Issues:
   - Ensure good lighting
   - Clear, front-facing images for enrollment
   - Multiple angles for better recognition

3. API Issues:
   - Verify your OpenAI API key
   - Check internet connection
   - Ensure sufficient API credits

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

