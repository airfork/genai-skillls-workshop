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
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        self.llm = ChatVertexAI(model="gemini-2.0-flash-exp", project=project_id, location=region)
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

    def handle_chat(self, user_message: str) -> str:
        """
        Handles a user's chat message by performing a similarity search and generating a response.
        Includes Model Armor sanitization for both input and output.

        Args:
            user_message: The user's input message

        Returns:
            The chatbot's response, or an error message if blocked/unavailable
        """
        if not self.retriever:
            return "Sorry, the knowledge base is currently unavailable."

        # Sanitize user prompt
        sanitized_prompt = self._sanitize_prompt(user_message)
        if sanitized_prompt is None:
            logger.warning("User prompt blocked - returning error message")
            return "I'm sorry, but I cannot process that request. Please rephrase your question."

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

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
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
