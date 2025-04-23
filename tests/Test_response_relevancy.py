import pytest
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import ResponseRelevancy

from utils import load_test_data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_data",
    load_test_data("Test_data.json"),
    indirect=True
)
async def test_response_relevancy(llm_wrapper, get_data):
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    response_relevancy = ResponseRelevancy(llm=llm_wrapper, embeddings=evaluator_embeddings)
    score = await response_relevancy.single_turn_ascore(get_data)
    print(score)
    assert score > 0.85
