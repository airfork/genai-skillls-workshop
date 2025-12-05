import os
import sys
import pytest
import csv
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatbot import Chatbot
import vertexai
from vertexai.generative_models import GenerativeModel


@pytest.fixture(scope="module")
def chatbot():
    """Initialize chatbot once for all tests."""
    return Chatbot()


@pytest.fixture(scope="module")
def evaluator_model():
    """Initialize Vertex AI model for evaluation."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    region = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    vertexai.init(project=project_id, location=region)
    return GenerativeModel("gemini-2.5-flash")


@pytest.fixture(scope="module")
def faq_data():
    """Load FAQ data from CSV."""
    faq_path = Path(__file__).parent.parent / "alaska-dept-of-snow" / "alaska-dept-of-snow-faqs.csv"
    faqs = []
    with open(faq_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            faqs.append({"question": row["question"], "answer": row["answer"]})
    return faqs


def evaluate_response(evaluator_model, question: str, expected_answer: str, actual_response: str) -> dict:
    """
    Use Vertex AI to evaluate if the chatbot response is informative and accurate.

    Returns:
        dict with keys: 'is_informative' (bool), 'score' (1-5), 'reasoning' (str)
    """
    evaluation_prompt = f"""You are an expert evaluator for chatbot responses. Evaluate the following chatbot response.

ORIGINAL QUESTION: {question}

EXPECTED INFORMATION (from knowledge base): {expected_answer}

ACTUAL CHATBOT RESPONSE: {actual_response}

Evaluate the response based on:
1. Does it contain the key information from the expected answer?
2. Is it informative and helpful?
3. Is it friendly and conversational?
4. Does it stay on topic?

Respond in the following JSON format ONLY (no other text):
{{
    "is_informative": true/false,
    "score": 1-5,
    "reasoning": "brief explanation"
}}

Score guide:
5 = Excellent - Contains all key info, friendly, accurate
4 = Good - Contains most key info, helpful
3 = Adequate - Contains some key info but missing details
2 = Poor - Vague or missing important information
1 = Unacceptable - Incorrect or unhelpful

Your JSON response:"""

    try:
        response = evaluator_model.generate_content(evaluation_prompt)
        result_text = response.text.strip()

        # Parse JSON from response
        import json

        # Handle markdown code blocks if present
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        evaluation = json.loads(result_text)
        return evaluation
    except Exception as e:
        # Fallback evaluation if parsing fails
        return {"is_informative": False, "score": 0, "reasoning": f"Evaluation failed: {str(e)}"}


class TestChatbotDirectQuestions:
    """Test chatbot with direct questions from FAQ."""

    def test_establishment_question(self, chatbot, evaluator_model, faq_data):
        """Test question about ADS establishment."""
        faq = next(f for f in faq_data if "established" in f["question"].lower())
        response = chatbot.handle_chat(faq["question"])

        evaluation = evaluate_response(evaluator_model, faq["question"], faq["answer"], response)

        print(f"\nQuestion: {faq['question']}")
        print(f"Response: {response}")
        print(f"Evaluation: {evaluation}")

        assert evaluation["is_informative"], f"Response not informative: {evaluation['reasoning']}"
        assert evaluation["score"] >= 3, f"Score too low ({evaluation['score']}): {evaluation['reasoning']}"

    def test_mission_question(self, chatbot, evaluator_model, faq_data):
        """Test question about ADS mission."""
        faq = next(f for f in faq_data if "mission" in f["question"].lower())
        response = chatbot.handle_chat(faq["question"])

        evaluation = evaluate_response(evaluator_model, faq["question"], faq["answer"], response)

        print(f"\nQuestion: {faq['question']}")
        print(f"Response: {response}")
        print(f"Evaluation: {evaluation}")

        assert evaluation["is_informative"], f"Response not informative: {evaluation['reasoning']}"
        assert evaluation["score"] >= 3, f"Score too low ({evaluation['score']}): {evaluation['reasoning']}"

    def test_school_closure_question(self, chatbot, evaluator_model, faq_data):
        """Test question about school closures."""
        faq = next(f for f in faq_data if "school closure" in f["question"].lower())
        response = chatbot.handle_chat(faq["question"])

        evaluation = evaluate_response(evaluator_model, faq["question"], faq["answer"], response)

        print(f"\nQuestion: {faq['question']}")
        print(f"Response: {response}")
        print(f"Evaluation: {evaluation}")

        assert evaluation["is_informative"], f"Response not informative: {evaluation['reasoning']}"
        assert evaluation["score"] >= 3, f"Score too low ({evaluation['score']}): {evaluation['reasoning']}"


class TestChatbotParaphrasedQuestions:
    """Test chatbot with paraphrased versions of FAQ questions."""

    def test_paraphrased_establishment(self, chatbot, evaluator_model, faq_data):
        """Test paraphrased question about establishment."""
        original_faq = next(f for f in faq_data if "established" in f["question"].lower())
        paraphrased_question = "When did the Alaska Department of Snow start?"

        response = chatbot.handle_chat(paraphrased_question)

        evaluation = evaluate_response(evaluator_model, paraphrased_question, original_faq["answer"], response)

        print(f"\nQuestion: {paraphrased_question}")
        print(f"Response: {response}")
        print(f"Evaluation: {evaluation}")

        assert evaluation["is_informative"], f"Response not informative: {evaluation['reasoning']}"
        assert evaluation["score"] >= 3, f"Score too low ({evaluation['score']}): {evaluation['reasoning']}"

    def test_paraphrased_contact(self, chatbot, evaluator_model, faq_data):
        """Test paraphrased question about contact info."""
        original_faq = next(f for f in faq_data if "unplowed road" in f["question"].lower())
        paraphrased_question = "How do I report a street that hasn't been plowed?"

        response = chatbot.handle_chat(paraphrased_question)

        evaluation = evaluate_response(evaluator_model, paraphrased_question, original_faq["answer"], response)

        print(f"\nQuestion: {paraphrased_question}")
        print(f"Response: {response}")
        print(f"Evaluation: {evaluation}")

        assert evaluation["is_informative"], f"Response not informative: {evaluation['reasoning']}"
        assert evaluation["score"] >= 3, f"Score too low ({evaluation['score']}): {evaluation['reasoning']}"

    def test_paraphrased_app(self, chatbot, evaluator_model, faq_data):
        """Test paraphrased question about mobile app."""
        original_faq = next(f for f in faq_data if "smartphone app" in f["question"].lower())
        paraphrased_question = "Is there a mobile app for tracking snow plows?"

        response = chatbot.handle_chat(paraphrased_question)

        evaluation = evaluate_response(evaluator_model, paraphrased_question, original_faq["answer"], response)

        print(f"\nQuestion: {paraphrased_question}")
        print(f"Response: {response}")
        print(f"Evaluation: {evaluation}")

        assert evaluation["is_informative"], f"Response not informative: {evaluation['reasoning']}"
        assert evaluation["score"] >= 3, f"Score too low ({evaluation['score']}): {evaluation['reasoning']}"

    def test_paraphrased_budget(self, chatbot, evaluator_model, faq_data):
        """Test paraphrased question about budget."""
        original_faq = next(f for f in faq_data if "CFO" in f["answer"] and "concerns" in f["question"].lower())
        paraphrased_question = "What are the CFO's worries about the department?"

        response = chatbot.handle_chat(paraphrased_question)

        evaluation = evaluate_response(evaluator_model, paraphrased_question, original_faq["answer"], response)

        print(f"\nQuestion: {paraphrased_question}")
        print(f"Response: {response}")
        print(f"Evaluation: {evaluation}")

        assert evaluation["is_informative"], f"Response not informative: {evaluation['reasoning']}"
        assert evaluation["score"] >= 3, f"Score too low ({evaluation['score']}): {evaluation['reasoning']}"


class TestChatbotResponseQuality:
    """Test overall response quality characteristics."""

    def test_response_is_friendly(self, chatbot):
        """Test that responses are friendly and conversational."""
        question = "What is the mission of ADS?"
        response = chatbot.handle_chat(question)

        # Response should not be empty and should be reasonably sized
        assert len(response) > 20, "Response too short"
        assert "error" not in response.lower() or "sorry" in response.lower()

    def test_response_not_making_up_info(self, chatbot, evaluator_model):
        """Test that chatbot doesn't make up information for unknown questions."""
        question = "What is the secret ingredient in ADS coffee?"
        response = chatbot.handle_chat(question)

        print(f"\nQuestion: {question}")
        print(f"Response: {response}")

        # Response should indicate lack of information
        evaluation_prompt = f"""Does this response appropriately indicate that the information is not available, rather than making up an answer?

Question: {question}
Response: {response}

Answer with just 'YES' or 'NO'."""

        eval_response = evaluator_model.generate_content(evaluation_prompt)
        result = eval_response.text.strip().upper()

        assert "YES" in result, "Chatbot appears to be making up information"

    def test_multiple_questions(self, chatbot, evaluator_model, faq_data):
        """Test asking multiple questions in sequence."""
        test_faqs = faq_data[:3]  # Test first 3 FAQs

        for faq in test_faqs:
            response = chatbot.handle_chat(faq["question"])
            evaluation = evaluate_response(evaluator_model, faq["question"], faq["answer"], response)

            print(f"\nQuestion: {faq['question']}")
            print(f"Response: {response}")
            print(f"Score: {evaluation['score']}")

            assert evaluation["is_informative"], f"Response not informative for '{faq['question']}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
