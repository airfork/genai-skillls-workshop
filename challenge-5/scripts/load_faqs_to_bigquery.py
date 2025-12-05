import os
import csv
import uuid
import logging
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from dotenv import load_dotenv
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
REGION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", "ads_data")
BIGQUERY_TABLE = "faq_knowledge_base"
FAQS_CSV_PATH = "alaska-dept-of-snow/alaska-dept-of-snow-faqs.csv"

def embed_text_to_vector(text: str) -> list[float]:
    """
    Generates an embedding for the given text using Vertex AI's TextEmbeddingModel.
    """
    try:
        vertexai.init(project=PROJECT_ID, location=REGION)
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
    except Exception as e:
        logger.error(f"Error generating embedding for text: {e}")
        return []

def load_faqs_to_bigquery():
    """
    Loads FAQs from a CSV file, generates embeddings, and inserts them into BigQuery.
    """
    if not PROJECT_ID:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set. Please set it in .env.")
        return
    if not os.path.exists(FAQS_CSV_PATH):
        logger.error(f"FAQ CSV file not found at {FAQS_CSV_PATH}. Aborting.")
        return

    bigquery_client = bigquery.Client(project=PROJECT_ID)
    table_ref = bigquery_client.dataset(BIGQUERY_DATASET).table(BIGQUERY_TABLE)

    try:
        bigquery_client.get_table(table_ref)
    except NotFound:
        logger.error(f"BigQuery table {BIGQUERY_DATASET}.{BIGQUERY_TABLE} not found.")
        logger.info("Please run scripts/setup_bigquery.py first to create the table.")
        return

    rows_to_insert = []
    with open(FAQS_CSV_PATH, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            question = row['question']
            answer = row['answer']
            
            # Generate embedding for the question+answer text
            # Combining question and answer for a more comprehensive embedding
            text_to_embed = f"Question: {question}\nAnswer: {answer}"
            embedding = embed_text_to_vector(text_to_embed)

            if not embedding:
                logger.warning(f"Skipping row {i+1} due to embedding failure.")
                continue

            rows_to_insert.append({
                "id": str(uuid.uuid4()), # Generate a unique ID for each FAQ
                "question": question,
                "answer": answer,
                "embedding": embedding,
                "category": "FAQ" # Default category
            })
    
    if not rows_to_insert:
        logger.info("No rows to insert.")
        return

    errors = bigquery_client.insert_rows_json(table_ref, rows_to_insert)

    if errors:
        logger.error(f"Errors occurred while inserting rows: {errors}")
    else:
        logger.info(f"Successfully inserted {len(rows_to_insert)} rows into BigQuery table {BIGQUERY_DATASET}.{BIGQUERY_TABLE}.")

if __name__ == "__main__":
    logger.info("Starting FAQ data loading to BigQuery...")
    load_faqs_to_bigquery()
    logger.info("FAQ data loading process completed.")
