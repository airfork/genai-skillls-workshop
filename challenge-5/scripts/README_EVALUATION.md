# Model Evaluation

This directory contains the model evaluation script that assesses the chatbot's performance using Vertex AI evaluation metrics.

## Overview

The evaluation script (`evaluate_model.py`) tests the chatbot against a set of FAQ questions and evaluates the responses using four key metrics:

### Evaluation Metrics

1. **Groundedness**: Measures whether responses are based on the provided context and don't contain hallucinated information
2. **Verbosity**: Assesses whether responses are appropriately concise and not overly verbose
3. **Instruction Following**: Evaluates how well responses follow the system instructions (friendly, conversational, accurate)
4. **Safety**: Checks whether responses are safe, appropriate, and don't contain harmful content

## Prerequisites

1. **Environment Setup**: Ensure all dependencies are installed
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Environment Variables**: Set up your `.env` file with:
   ```
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_REGION=us-central1
   BIGQUERY_DATASET=ads_data
   GOOGLE_MAPS_API_KEY=your-maps-api-key
   ```

3. **BigQuery Data**: Ensure the FAQ knowledge base is loaded
   ```bash
   uv run scripts/setup_bigquery.py
   uv run scripts/load_faqs_to_bigquery.py
   ```

4. **Vertex AI Permissions**: Your GCP account needs access to:
   - Vertex AI Evaluation API
   - BigQuery
   - Model Armor (if using sanitization)

## Running the Evaluation

### Basic Usage

Run the evaluation script:

```bash
uv run scripts/evaluate_model.py
```

### What It Does

1. **Initializes**: Sets up the chatbot and Vertex AI
2. **Generates Responses**: Sends 5 test queries to the chatbot
3. **Evaluates**: Runs Vertex AI evaluation metrics on the responses
4. **Reports**: Prints results to console and saves to `evaluation_results.txt`

### Expected Output

```
INFO:__main__:Starting chatbot evaluation...
INFO:__main__:Initialized Vertex AI with project: your-project, region: us-central1
INFO:__main__:Initializing chatbot...
INFO:__main__:Created evaluation dataset with 5 examples
INFO:__main__:Generating chatbot responses for evaluation...
INFO:__main__:Prompt: When was the Alaska Department of Snow established?...
INFO:__main__:Response: The Alaska Department of Snow was established in 1959...
...
INFO:__main__:Running evaluation task...
================================================================================
EVALUATION RESULTS
================================================================================

--- Summary Metrics ---
groundedness/mean: 0.9500
verbosity/mean: 0.8200
instruction_following/mean: 0.9100
safety/mean: 1.0000

--- Detailed Metrics Table ---
[Table with per-example metrics]

================================================================================
INFO:__main__:Results saved to evaluation_results.txt
INFO:__main__:Evaluation completed successfully!
```

## Understanding the Results

### Metric Scores

Each metric is scored from 0 to 1 (or 1 to 5 depending on the metric):
- **0.9-1.0**: Excellent
- **0.8-0.9**: Good
- **0.7-0.8**: Acceptable
- **<0.7**: Needs improvement

### Results File

The evaluation creates `evaluation_results.txt` in the project root with:
- Summary metrics (mean scores across all examples)
- Detailed metrics table (per-example scores)

## Customizing the Evaluation

### Adding More Test Cases

Edit `create_evaluation_dataset()` in `evaluate_model.py`:

```python
def create_evaluation_dataset():
    return [
        {
            "prompt": "Your test question",
            "reference": "Expected answer",
            "context": "Knowledge base context",
        },
        # Add more examples...
    ]
```

### Adding Different Metrics

Import and add other metrics:

```python
from vertexai.evaluation import MetricPromptTemplateExamples

# Add custom metrics
fluency_metric = PointwiseMetric(
    metric="fluency",
    metric_prompt_template=MetricPromptTemplateExamples.Pointwise.FLUENCY,
)

# Add to evaluation task
eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[groundedness_metric, verbosity_metric, fluency_metric],
)
```

### Changing Evaluation Dataset Size

Modify the dataset in `create_evaluation_dataset()` to include more or fewer examples. More examples provide better statistical significance.

## Troubleshooting

### "Evaluation API not enabled"

Enable the Vertex AI Evaluation API:
```bash
gcloud services enable aiplatform.googleapis.com
```

### "Permission denied"

Ensure your GCP account has the following roles:
- Vertex AI User
- BigQuery Data Viewer
- Model Armor User (if using sanitization)

### "Chatbot initialization failed"

Check that:
- BigQuery table exists and has data
- Environment variables are set correctly
- You have network access to GCP services

### Low metric scores

If scores are consistently low:
1. Review the chatbot prompt template
2. Check if the knowledge base has relevant information
3. Verify responses are being generated correctly
4. Consider adjusting the system instructions

## Integration with CI/CD

You can integrate this into your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run model evaluation
  env:
    GOOGLE_CLOUD_PROJECT: ${{ secrets.GCP_PROJECT_ID }}
  run: |
    uv run scripts/evaluate_model.py

- name: Check evaluation results
  run: |
    # Parse results and fail if scores are below threshold
    python scripts/check_eval_thresholds.py
```

## Additional Resources

- [Vertex AI Evaluation Documentation](https://cloud.google.com/vertex-ai/docs/evaluation/introduction)
- [Evaluation Metrics Reference](https://cloud.google.com/vertex-ai/docs/evaluation/metrics)
- [Best Practices for LLM Evaluation](https://cloud.google.com/vertex-ai/docs/evaluation/best-practices)
