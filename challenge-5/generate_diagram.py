#!/usr/bin/env python3
"""
Generate GCP architecture diagram for the application.
Requires: pip install diagrams
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.gcp.analytics import BigQuery
from diagrams.gcp.compute import Run
from diagrams.gcp.ml import AIPlatform
from diagrams.onprem.client import Users
from diagrams.programming.framework import FastAPI
from diagrams.onprem.network import Internet

# Optional: use graphviz attributes for styling
graph_attr = {
    "fontsize": "14",
    "bgcolor": "white"
}

with Diagram("GenAI Skills Workshop - Challenge 5 Architecture",
             show=False,
             direction="LR",
             graph_attr=graph_attr,
             filename="architecture_diagram"):

    user = Users("User")

    with Cluster("Data Ingestion Pipeline"):
        bq_raw = BigQuery("Raw Data\n(CSV Import)")
        bq_embed = BigQuery("BigQuery Table\nwith Embeddings")
        bq_raw >> Edge(label="embed & load") >> bq_embed

    with Cluster("Cloud Run Application"):
        app = Run("Cloud Run\nService")
        api = FastAPI("FastAPI\nBackend")
        app - api

    with Cluster("External Services"):
        weather = Internet("Weather\nAPI")
        gmaps = Internet("Google Maps\nAPI")
        vertex = AIPlatform("Vertex AI\n(Agent/LLM)")

    # User interactions
    user >> Edge(label="HTTP requests") >> app

    # App reads from BQ
    api >> Edge(label="query embeddings") >> bq_embed

    # App calls external services
    api >> Edge(label="weather data") >> weather
    api >> Edge(label="location data") >> gmaps
    api >> Edge(label="LLM calls") >> vertex
