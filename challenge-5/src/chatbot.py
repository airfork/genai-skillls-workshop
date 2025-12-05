import logging
import os
from importlib import import_module
from typing import Optional, Type

import vertexai
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
    """

    def __init__(self):
        """
        Initializes the Chatbot by setting up the language model, embeddings, and vector store.
        """
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        region = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        vertexai.init(project=project_id, location=region)

        self.llm = ChatVertexAI(model="gemini-2.0-flash-exp", project=project_id, location=region)
        self.embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project=project_id, location=region)

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

    def handle_chat(self, user_message: str) -> str:
        """
        Handles a user's chat message by performing a similarity search and generating a response.
        """
        if not self.retriever:
            return "Sorry, the knowledge base is currently unavailable."

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
            response = rag_chain.invoke(user_message)
            logger.info(f"Chatbot interaction - Prompt: {user_message} | Response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error invoking RAG chain: {e}")
            return "Sorry, I encountered an error while processing your request."
