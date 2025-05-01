import json
import os
from pathlib import Path

import pytest_html
import requests


def load_test_data(filename):
    project_directory = Path(__file__).parent.absolute()
    test_data_path = project_directory / "test_data" / filename
    with open(test_data_path) as f:
        return json.load(f)


def get_llm_response(test_data):
    response_dict = requests.post(os.getenv("ONSITEAI_API_BASE"),
                                  headers={"Content-Type": "application/json", "x-apikey": "openai"},
                                  json={
                                      "message": {"message": test_data["message"],
                                                  "metadata": {
                                                      "additional_context": {
                                                      }
                                                  }
                                                  },
                                      "answering_agent": "LearnAIAgent",
                                      "formatting": "json"
                                  }).json()
    return response_dict


def attach_to_html_report(question, contexts, message, reference, score, request):
    extra = getattr(request.node, "_extra", [])
    extra.append(pytest_html.extras.text(f"Prompt: {question}", name="Prompt"))
    extra.append(pytest_html.extras.text(f"Context: {contexts}", name="Context"))
    extra.append(pytest_html.extras.text(f"Message: {message}", name="Message"))
    extra.append(pytest_html.extras.text(f"Score: {score}", name="RAGAS Score"))
    extra.append(pytest_html.extras.text(f"Reference: {reference}", name="Reference"))
    request.node._extra = extra
