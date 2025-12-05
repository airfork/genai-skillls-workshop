import os
import uuid
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import logging
from .weather_service import get_alaska_weather
from .chatbot import Chatbot

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# Store conversation histories per session
conversation_histories = {}

# Initialize the chatbot
try:
    chat_service = Chatbot()
    logger.info("Chatbot initialized successfully.")
except Exception as e:
    chat_service = None
    logger.error(f"Failed to initialize chatbot. Chat will be disabled. Error: {e}")


@app.route("/")
def index():
    """
    Render the main page, showing the forecast for the 5 most popular cities in Alaska.
    """
    weather_data = get_alaska_weather()
    return render_template("index.html", weather_data=weather_data)


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Endpoint for the chatbot with conversation history support.
    """
    if not chat_service:
        return jsonify({"error": "Chat service is not available."}), 503

    # Get or create session ID
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    session_id = session["session_id"]

    # Initialize conversation history for new sessions
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    data = request.get_json()
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Get conversation history for this session
    history = conversation_histories[session_id]

    # Handle the chat with history
    response_text = chat_service.handle_chat(user_message, history)

    # Update conversation history
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response_text})

    # Keep only last 10 exchanges (20 messages) to avoid memory issues
    if len(history) > 20:
        conversation_histories[session_id] = history[-20:]

    return jsonify({"response": response_text})


@app.route("/api/chat/reset", methods=["POST"])
def reset_chat():
    """
    Reset conversation history for the current session.
    """
    if "session_id" in session:
        session_id = session["session_id"]
        if session_id in conversation_histories:
            del conversation_histories[session_id]
    return jsonify({"status": "reset"})


@app.route("/health")
def health():
    """Health check endpoint for Cloud Run."""
    return "OK", 200
