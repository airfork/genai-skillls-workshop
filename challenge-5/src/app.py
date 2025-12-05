import os
from flask import Flask, render_template, request, jsonify
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
    Endpoint for the chatbot.
    """
    if not chat_service:
        return jsonify({"error": "Chat service is not available."}), 503

    data = request.get_json()
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response_text = chat_service.handle_chat(user_message)
    return jsonify({"response": response_text})


@app.route("/health")
def health():
    """Health check endpoint for Cloud Run."""
    return "OK", 200
