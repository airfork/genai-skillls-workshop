from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_bigquery_resources():
    """
    Creates the BigQuery dataset and table for the ADS Knowledge Base.
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset_id = os.environ.get("BIGQUERY_DATASET", "ads_data")
    table_id = "faq_knowledge_base"
    
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set.")
        return

    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)

    # 1. Create Dataset
    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_id} already exists.")
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        dataset = client.create_dataset(dataset)
        logger.info(f"Created dataset {client.project}.{dataset.dataset_id}")

    # 2. Create Table
    table_ref = dataset_ref.table(table_id)
    
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("question", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("answer", "STRING", mode="REQUIRED"),
        # For Vector Search (Vertex AI Vector Search or BigQuery Vector Search)
        # We will store embeddings here.
        # Dimensions depends on the model (e.g., gecko=768). 
        # Using REPEATED FLOAT for embedding vector.
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"), 
        bigquery.SchemaField("category", "STRING", mode="NULLABLE"),
    ]

    try:
        client.get_table(table_ref)
        logger.info(f"Table {table_id} already exists.")
    except NotFound:
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table)
        logger.info(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")

if __name__ == "__main__":
    # Ensure .env is loaded if running locally
    from dotenv import load_dotenv
    load_dotenv()
    
    create_bigquery_resources()
