"""
Model Evaluation Script

This script evaluates the chatbot using Vertex AI's evaluation metrics including:
- Groundedness: Whether responses are based on provided context
- Verbosity: Whether responses are appropriately concise
- Instruction Following: Whether responses follow system instructions
- Safety: Whether responses are safe and appropriate
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.evaluation import (
    EvalTask,
    MetricPromptTemplateExamples,
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
)

from chatbot import Chatbot

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_vertex_ai():
    """Initialize Vertex AI with project settings."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    region = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")

    aiplatform.init(project=project_id, location=region)
    logger.info(f"Initialized Vertex AI with project: {project_id}, region: {region}")


def create_evaluation_dataset():
    """
    Create evaluation dataset with test queries and expected behaviors.

    Returns:
        List of evaluation examples
    """
    return [
        {
            "prompt": "When was the Alaska Department of Snow established?",
            "reference": "The Alaska Department of Snow was established in 1959",
            "context": "The Alaska Department of Snow (ADS) was established in 1959, coinciding with Alaska's admission as a U.S. state.",
        },
        {
            "prompt": "What is the mission of ADS?",
            "reference": "The mission is to ensure safe, efficient travel and infrastructure continuity by coordinating snow removal services across Alaska.",
            "context": "Our mission is to ensure safe, efficient travel and infrastructure continuity by coordinating snow removal services across the state's 650,000 square miles.",
        },
        {
            "prompt": "Who do I contact to report an unplowed road?",
            "reference": "Contact your local ADS regional office which maintains a hotline for snow-related service requests.",
            "context": "Contact your local ADS regional office. Each region maintains a hotline for snow-related service requests and emergencies.",
        },
        {
            "prompt": "Does ADS have a smartphone app?",
            "reference": "Yes, the ADS SnowLine app offers real-time plow tracking, road conditions, and service requests.",
            "context": "Yes. The ADS 'SnowLine' app offers real-time plow tracking, road conditions, and the ability to submit service requests directly from your phone.",
        },
        {
            "prompt": "How many people does ADS serve?",
            "reference": "ADS serves approximately 750,000 people across Alaska.",
            "context": "ADS serves approximately 750,000 people across Alaska's widely distributed communities and remote areas.",
        },
    ]


def generate_responses(chatbot, evaluation_data):
    """
    Generate chatbot responses for evaluation dataset.

    Args:
        chatbot: Initialized Chatbot instance
        evaluation_data: List of evaluation examples

    Returns:
        List of evaluation examples with responses
    """
    logger.info("Generating chatbot responses for evaluation...")

    for example in evaluation_data:
        try:
            response = chatbot.handle_chat(example["prompt"])
            example["response"] = response
            logger.info(f"Prompt: {example['prompt'][:50]}...")
            logger.info(f"Response: {response[:100]}...")
        except Exception as e:
            logger.error(f"Error generating response for prompt '{example['prompt']}': {e}")
            example["response"] = "Error generating response"

    return evaluation_data


def evaluate_with_metrics(evaluation_data):
    """
    Evaluate responses using Vertex AI metrics.

    Args:
        evaluation_data: List of evaluation examples with responses

    Returns:
        Evaluation results
    """
    logger.info("Starting evaluation with Vertex AI metrics...")

    # Define metrics
    groundedness_metric = PointwiseMetric(
        metric="groundedness",
        metric_prompt_template=MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
    )

    verbosity_metric = PointwiseMetric(
        metric="verbosity",
        metric_prompt_template=MetricPromptTemplateExamples.Pointwise.VERBOSITY,
    )

    instruction_following_metric = PointwiseMetric(
        metric="instruction_following",
        metric_prompt_template=MetricPromptTemplateExamples.Pointwise.INSTRUCTION_FOLLOWING,
    )

    safety_metric = PointwiseMetric(
        metric="safety",
        metric_prompt_template=MetricPromptTemplateExamples.Pointwise.SAFETY,
    )

    # Prepare evaluation data in required format
    eval_dataset = {
        "prompt": [ex["prompt"] for ex in evaluation_data],
        "response": [ex["response"] for ex in evaluation_data],
        "context": [ex["context"] for ex in evaluation_data],
        "reference": [ex["reference"] for ex in evaluation_data],
    }

    # Create evaluation task
    eval_task = EvalTask(
        dataset=eval_dataset,
        metrics=[groundedness_metric, verbosity_metric, instruction_following_metric, safety_metric],
    )

    # Run evaluation
    logger.info("Running evaluation task...")
    result = eval_task.evaluate()

    return result


def print_evaluation_results(result):
    """
    Print evaluation results in a readable format.

    Args:
        result: Evaluation result object
    """
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    # Print summary metrics
    summary_metrics = result.summary_metrics
    logger.info("\n--- Summary Metrics ---")
    for metric_name, metric_value in summary_metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")

    # Print metrics table if available
    if hasattr(result, "metrics_table"):
        logger.info("\n--- Detailed Metrics Table ---")
        logger.info(result.metrics_table)

    logger.info("\n" + "=" * 80)


def save_evaluation_results(result, output_file="evaluation_results.txt"):
    """
    Save evaluation results to a file.

    Args:
        result: Evaluation result object
        output_file: Output file name
    """
    output_path = Path(__file__).parent.parent / output_file

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CHATBOT EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("--- Summary Metrics ---\n")
        for metric_name, metric_value in result.summary_metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")

        if hasattr(result, "metrics_table"):
            f.write("\n--- Detailed Metrics Table ---\n")
            f.write(str(result.metrics_table))
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"Results saved to {output_path}")


def main():
    """Main evaluation workflow."""
    try:
        # Initialize
        logger.info("Starting chatbot evaluation...")
        initialize_vertex_ai()

        # Initialize chatbot
        logger.info("Initializing chatbot...")
        chatbot = Chatbot()

        # Create evaluation dataset
        evaluation_data = create_evaluation_dataset()
        logger.info(f"Created evaluation dataset with {len(evaluation_data)} examples")

        # Generate responses
        evaluation_data = generate_responses(chatbot, evaluation_data)

        # Run evaluation
        result = evaluate_with_metrics(evaluation_data)

        # Print and save results
        print_evaluation_results(result)
        save_evaluation_results(result)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
