import pytest
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv

from utils import get_llm_response

load_dotenv()  # Load environment variables from .env file


@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    return lang_chain_llm


@pytest.fixture
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)
    referenced_docs = response_dict.get("intermediate_steps")[0][0][3]["referenced_docs"]

    sample = SingleTurnSample(
        user_input=test_data["message"],
        response=response_dict["message"],
        retrieved_contexts=[doc["text"] for doc in referenced_docs],
        reference=test_data["reference"]
    )
    return {
        "sample": sample,
        "question": test_data["message"],
        "contexts": [doc["text"] for doc in referenced_docs],
        "message": response_dict["message"],
        "reference": test_data["reference"],
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if hasattr(item, "_extra"):
        rep.extra = getattr(item, "_extra")
