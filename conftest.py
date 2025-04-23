import pytest
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv

from utils import get_llm_response

load_dotenv()  # Load environment variables from .env file


@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    return lang_chain_llm


@pytest.fixture
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)
    referenced_docs = response_dict.get("intermediate_steps")[0][0][3]["referenced_docs"]
    # print([referenced_docs[0]["text"]])
    # print(response_dict["message"])

    sample = SingleTurnSample(
        user_input=test_data["message"],
        response=response_dict["message"],
        retrieved_contexts=[doc["text"] for doc in referenced_docs],
        # reference=test_data["reference"]
    )
    return sample
