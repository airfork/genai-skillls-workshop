# Project Overview

This project is a web-based chatbot for the Alaska Department of Snow (ADS). The application provides weather forecasts for major Alaskan cities and answers frequently asked questions (FAQs) using a Retrieval-Augmented Generation (RAG) approach with Google Cloud's BigQuery and Vertex AI.

## Technologies

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Data Storage & RAG:** Google BigQuery
- **AI & Embeddings:** Google Vertex AI (`text-embedding-004`)
- **Dependency Management:** `uv`

## Project Structure

- `src/app.py`: The main Flask application file.
- `src/weather_service.py`: A service to fetch weather data from the National Weather Service (NWS) API.
- `scripts/setup_bigquery.py`: A script to set up the BigQuery dataset and table.
- `scripts/load_faqs_to_bigquery.py`: A script to load FAQ data from a CSV file into BigQuery and generate embeddings.
- `alaska-dept-of-snow/`: Contains the raw FAQ data.

## Building and Running

### 1. Set up the Environment

- Install dependencies: `uv pip install -r requirements.txt`
- Set up a `.env` file with the following environment variables:
  - `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID.
  - `GOOGLE_CLOUD_REGION`: The Google Cloud region to use (e.g., `us-central1`).
  - `GOOGLE_MAPS_API_KEY`: Your Google Maps Geocoding API key.
  - `BIGQUERY_DATASET`: The name of the BigQuery dataset (defaults to `ads_data`).

### 2. Set up the Backend

- Run the BigQuery setup script: `uv run scripts/setup_bigquery.py`
- Load the FAQ data into BigQuery: `uv run scripts/load_faqs_to_bigquery.py`

### 3. Run the Application

- Start the Flask development server: `uv run src/app.py`

## Development Conventions

- The application uses `uv` for dependency management.
- The backend is built with Flask.
- The RAG functionality is intended to be implemented in `src/app.py` using LangChain, BigQuery, and Vertex AI.
