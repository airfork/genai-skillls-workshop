import json
import logging
import os
from importlib import import_module
from typing import Optional, Type

import vertexai
from google.cloud import modelarmor_v1
from google.cloud.modelarmor_v1 import types as armor_types
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from .weather_service import get_weather_forecast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def get_city_weather(city_name: str) -> str:
    """
    Get current weather information for any city.

    This tool fetches real-time weather data including temperature and conditions.
    It also detects whether the city is in Alaska or not.

    Args:
        city_name: The name of the city to get weather for (e.g., "Anchorage", "Seattle", "New York")

    Returns:
        A JSON string with weather information including whether the city is in Alaska
    """
    try:
        weather_data = get_weather_forecast(city_name, check_alaska=True)
        return json.dumps(weather_data)
    except Exception as e:
        logger.error(f"Error fetching weather for {city_name}: {e}")
        return json.dumps(
            {"city": city_name, "temp": "--", "condition": "Error fetching weather", "is_in_alaska": False}
        )


def _load_bigquery_vector_store():
    """
    Try multiple import paths/names to find the BigQuery vector store across
    LangChain / Google package versions.
    """
    candidates = [
        ("langchain_google_community.vectorstores.bigquery", ["BigQueryVectorStore", "BigQueryVectorSearch"]),
        ("langchain_google_community.vectorstores", ["BigQueryVectorStore", "BigQueryVectorSearch"]),
        ("langchain_community.vectorstores.bigquery", ["BigQueryVectorStore", "BigQueryVectorSearch"]),
        ("langchain_community.vectorstores", ["BigQueryVectorStore", "BigQueryVectorSearch"]),
    ]
    for module_path, names in candidates:
        try:
            mod = import_module(module_path)
            for name in names:
                cls = getattr(mod, name, None)
                if cls:
                    return cls
        except Exception:
            continue
    return None


class Chatbot:
    """
    A chatbot class that uses LangChain, BigQuery, and Vertex AI to answer questions.
    Includes Model Armor for request/response sanitization.
    """

    def __init__(self):
        """
        Initializes the Chatbot by setting up the language model, embeddings, vector store, and Model Armor.
        """
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        region = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        vertexai.init(project=project_id, location=region)

        # Bind tools to the LLM
        self.llm = ChatVertexAI(model="gemini-2.0-flash-exp", project=project_id, location=region).bind_tools(
            [get_city_weather]
        )
        self.llm_without_tools = ChatVertexAI(model="gemini-2.0-flash-exp", project=project_id, location=region)
        self.embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project=project_id, location=region)

        # Initialize Model Armor client
        self.armor_client = None
        self.armor_template_path = f"projects/{project_id}/locations/us/templates/challenge-1-template"
        try:
            self.armor_client = modelarmor_v1.ModelArmorClient(
                client_options={"api_endpoint": "modelarmor.us.rep.googleapis.com"}
            )
            logger.info("Model Armor client initialized successfully")
        except Exception as e:
            logger.warning("Failed to initialize Model Armor client: %s", e)

        self.retriever = None
        BQVectorStore: Optional[Type] = _load_bigquery_vector_store()
        if BQVectorStore:
            try:
                dataset_id = os.environ.get("BIGQUERY_DATASET", "ads_data")
                table_id = "faq_knowledge_base"

                vector_store = BQVectorStore(
                    project_id=project_id,
                    dataset_name=dataset_id,
                    table_name=table_id,
                    location="US",
                    embedding=self.embeddings,
                    content_field="answer",
                    text_embedding_field="embedding",
                )
                self.retriever = vector_store.as_retriever()
            except Exception as e:
                logger.warning("Failed to initialize BigQuery vector store: %s", e)
        else:
            logger.warning("BigQuery vector store class not found in installed packages. RAG will be disabled.")

    def _sanitize_prompt(self, user_prompt: str) -> Optional[str]:
        """
        Sanitize user prompt using Model Armor.

        Args:
            user_prompt: The user's input prompt

        Returns:
            The sanitized prompt if safe, None if blocked
        """
        if not self.armor_client:
            logger.warning("Model Armor not available, skipping prompt sanitization")
            return user_prompt

        try:
            user_prompt_data = modelarmor_v1.DataItem(text=user_prompt)
            request = modelarmor_v1.SanitizeUserPromptRequest(
                name=self.armor_template_path,
                user_prompt_data=user_prompt_data,
            )

            response = self.armor_client.sanitize_user_prompt(request=request)

            if response.sanitization_result.filter_match_state == armor_types.FilterMatchState.MATCH_FOUND:
                logger.warning("User prompt blocked by Model Armor: %s", user_prompt[:50])
                return None

            logger.debug("User prompt passed Model Armor sanitization")
            return user_prompt

        except Exception as e:
            logger.error("Error sanitizing prompt with Model Armor: %s", e)
            # Fail open - allow the prompt if sanitization fails
            return user_prompt

    def _sanitize_response(self, model_response: str) -> Optional[str]:
        """
        Sanitize model response using Model Armor.

        Args:
            model_response: The model's generated response

        Returns:
            The sanitized response if safe, None if blocked
        """
        if not self.armor_client:
            logger.warning("Model Armor not available, skipping response sanitization")
            return model_response

        try:
            model_response_data = modelarmor_v1.DataItem(text=model_response)
            request = modelarmor_v1.SanitizeModelResponseRequest(
                name=self.armor_template_path,
                model_response_data=model_response_data,
            )

            response = self.armor_client.sanitize_model_response(request=request)

            if response.sanitization_result.filter_match_state == armor_types.FilterMatchState.MATCH_FOUND:
                logger.warning("Model response blocked by Model Armor: %s", model_response[:50])
                return None

            logger.debug("Model response passed Model Armor sanitization")
            return model_response

        except Exception as e:
            logger.error("Error sanitizing response with Model Armor: %s", e)
            # Fail open - allow the response if sanitization fails
            return model_response

    def _handle_weather_query(self, user_message: str, conversation_history: list = None) -> Optional[str]:
        """
        Check if the user is asking about weather and handle it with tools.

        Args:
            user_message: The user's input message
            conversation_history: Previous conversation messages for context

        Returns:
            Weather response if it's a weather query, None otherwise
        """
        # Build context from conversation history
        history_context = ""
        if conversation_history:
            recent_history = conversation_history[-6:]  # Last 3 exchanges
            for msg in recent_history:
                role = msg["role"].capitalize()
                history_context += f"{role}: {msg['content']}\n"

        # Use the LLM with tools to detect and handle weather queries
        weather_prompt = f"""You are a helpful assistant for the Alaska Department of Snow.

{history_context}
If the user is asking about weather for a specific city, use the get_city_weather tool to fetch the current weather.
If the user refers to a city with words like "what about", "how about", or "there", use the conversation history to understand which city they mean.

After getting the weather data:
- If the city is in Alaska, respond with the weather in a friendly tone
- If the city is NOT in Alaska, acknowledge it's not in Alaska but still provide the weather, saying something like "That's not in Alaska, but the weather in [city] is..."

User question: {user_message}"""

        try:
            # Invoke the LLM with tools
            response = self.llm.invoke(weather_prompt)

            # Check if tool was called
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Execute the tool
                tool_call = response.tool_calls[0]
                tool_output = get_city_weather.invoke(tool_call["args"])

                # Parse the weather data
                weather_data = json.loads(tool_output)

                # Generate a friendly response based on Alaska status
                if weather_data.get("is_in_alaska"):
                    final_response = f"The current weather in {weather_data['city']}, Alaska is {weather_data['temp']} with {weather_data['condition'].lower()}."
                else:
                    final_response = f"That's not in Alaska, but the weather in {weather_data['full_location']} is {weather_data['temp']} with {weather_data['condition'].lower()}."

                return final_response

            # Check if the response content indicates it's a weather query
            response_text = response.content if hasattr(response, "content") else str(response)
            if "weather" in user_message.lower() and ("temperature" in response_text.lower() or "°" in response_text):
                return response_text

            return None

        except Exception as e:
            logger.error(f"Error handling weather query: {e}")
            return None

    def handle_chat(self, user_message: str, conversation_history: list = None) -> str:
        """
        Handles a user's chat message by performing a similarity search and generating a response.
        Includes Model Armor sanitization, tool calling for weather queries, and conversation history.

        Args:
            user_message: The user's input message
            conversation_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]

        Returns:
            The chatbot's response, or an error message if blocked/unavailable
        """
        if not self.retriever:
            return "Sorry, the knowledge base is currently unavailable."

        if conversation_history is None:
            conversation_history = []

        # Sanitize user prompt
        sanitized_prompt = self._sanitize_prompt(user_message)
        if sanitized_prompt is None:
            logger.warning("User prompt blocked - returning error message")
            return "I'm sorry, but I cannot process that request. Please rephrase your question."

        # Check if this is a weather query or follow-up
        # Look for weather keywords or follow-up patterns
        is_weather_related = "weather" in sanitized_prompt.lower() or any(
            pattern in sanitized_prompt.lower()
            for pattern in ["what about", "how about", "what's it like in", "temperature in", "forecast for"]
        )

        # Also check if previous message was about weather
        if conversation_history and len(conversation_history) > 0:
            last_exchange = conversation_history[-2:]  # Last user + assistant message
            for msg in last_exchange:
                if "weather" in msg["content"].lower() or "temperature" in msg["content"].lower():
                    is_weather_related = True
                    break

        if is_weather_related:
            weather_response = self._handle_weather_query(sanitized_prompt, conversation_history)
            if weather_response:
                # Sanitize the weather response
                sanitized_response = self._sanitize_response(weather_response)
                if sanitized_response is None:
                    logger.warning("Weather response blocked - returning safe message")
                    return "I apologize, but I cannot provide that information. Please contact your local ADS office for assistance."

                logger.info(f"Chatbot interaction (weather) - Prompt: {user_message} | Response: {sanitized_response}")
                return sanitized_response

        # Fall back to RAG for non-weather queries
        # Build conversation context
        history_text = ""
        if conversation_history:
            recent_history = conversation_history[-4:]  # Last 2 exchanges
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        prompt_template = """
        You are a friendly and helpful AI assistant for the Alaska Department of Snow.
        Your goal is to provide warm, conversational responses while answering questions accurately based on the provided context.

        Guidelines:
        - Be friendly, welcoming, and empathetic in your tone
        - Use a conversational style that makes people feel comfortable
        - If the context contains the answer, provide it in a helpful and clear way
        - If the context doesn't contain the answer, politely let them know you don't have that specific information
        - Never make up information - only use what's in the context
        - Feel free to acknowledge Alaska's unique challenges (weather, distance, etc.) when relevant
        - Use the conversation history to understand follow-up questions

        CONVERSATION HISTORY:
        {history}

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["history", "context", "question"]
        )

        rag_chain = (
            {
                "history": lambda _: history_text,
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm_without_tools
            | StrOutputParser()
        )

        try:
            response = rag_chain.invoke(sanitized_prompt)

            # Sanitize model response
            sanitized_response = self._sanitize_response(response)
            if sanitized_response is None:
                logger.warning("Model response blocked - returning safe message")
                return "I apologize, but I cannot provide that information. Please contact your local ADS office for assistance."

            logger.info(f"Chatbot interaction - Prompt: {user_message} | Response: {sanitized_response}")
            return sanitized_response
        except Exception as e:
            logger.error(f"Error invoking RAG chain: {e}")
            return "Sorry, I encountered an error while processing your request."
