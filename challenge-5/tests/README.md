# Chatbot Tests

This directory contains comprehensive tests for the Alaska Department of Snow chatbot.

## Test Structure

### `test_chatbot.py`

The test suite includes three main test classes:

1. **TestChatbotDirectQuestions**
   - Tests chatbot with exact questions from the FAQ CSV
   - Validates responses contain accurate information
   - Tests: establishment, mission, school closures

2. **TestChatbotParaphrasedQuestions**
   - Tests chatbot with paraphrased/reworded versions of FAQ questions
   - Ensures the chatbot can understand different phrasings
   - Tests: establishment (paraphrased), contact info, mobile app, budget concerns

3. **TestChatbotResponseQuality**
   - Tests overall response characteristics
   - Validates friendliness and conversational tone
   - Ensures chatbot doesn't make up information
   - Tests multiple questions in sequence

## Evaluation Method

Tests use **Vertex AI's Gemini model** as an evaluator to assess:
- Whether responses contain key information from the knowledge base
- If responses are informative and helpful
- If responses are friendly and conversational
- If responses stay on topic

Each response is scored 1-5:
- **5**: Excellent - Contains all key info, friendly, accurate
- **4**: Good - Contains most key info, helpful
- **3**: Adequate - Contains some key info but missing details
- **2**: Poor - Vague or missing important information
- **1**: Unacceptable - Incorrect or unhelpful

Tests pass if responses are informative and score ≥ 3.

## Running Tests

### Prerequisites

1. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

2. Ensure environment variables are set in `.env`:
   ```
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_REGION=us-central1
   BIGQUERY_DATASET=ads_data
   ```

3. Ensure the BigQuery table is populated with FAQ data:
   ```bash
   uv run scripts/load_faqs_to_bigquery.py
   ```

### Run All Tests

```bash
uv run pytest tests/test_chatbot.py -v
```

### Run Specific Test Class

```bash
# Test only direct questions
uv run pytest tests/test_chatbot.py::TestChatbotDirectQuestions -v

# Test only paraphrased questions
uv run pytest tests/test_chatbot.py::TestChatbotParaphrasedQuestions -v

# Test only response quality
uv run pytest tests/test_chatbot.py::TestChatbotResponseQuality -v
```

### Run Specific Test

```bash
uv run pytest tests/test_chatbot.py::TestChatbotDirectQuestions::test_establishment_question -v
```

### Run with Detailed Output

```bash
uv run pytest tests/test_chatbot.py -v -s
```

The `-s` flag shows print statements, which display:
- Questions asked
- Chatbot responses
- Evaluation scores and reasoning

## Test Output Example

```
tests/test_chatbot.py::TestChatbotDirectQuestions::test_establishment_question

Question: When was the Alaska Department of Snow established?
Response: The Alaska Department of Snow was established in 1959, when Alaska became a state!
Evaluation: {'is_informative': True, 'score': 5, 'reasoning': 'Contains accurate information, friendly tone'}

PASSED
```

## CI/CD Integration

These tests can be integrated into your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run chatbot tests
  env:
    GOOGLE_CLOUD_PROJECT: ${{ secrets.GCP_PROJECT_ID }}
  run: |
    uv run pytest tests/test_chatbot.py -v
```

## Troubleshooting

### Tests fail with "knowledge base is currently unavailable"

- Check that BigQuery table `ads_data.faq_knowledge_base` exists and contains data
- Verify environment variables are set correctly
- Ensure GCP credentials are configured

### Evaluation model errors

- Verify you have access to Vertex AI APIs
- Check that `gemini-2.0-flash-exp` model is available in your region
- Review GCP quota limits

### Import errors

- Ensure all dependencies are installed: `uv pip install -r requirements.txt`
- Check that `src/chatbot.py` exists and is error-free
