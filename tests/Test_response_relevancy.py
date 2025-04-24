import pytest
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import ResponseRelevancy

from utils import load_test_data, attach_to_html_report


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_data",
    load_test_data("Test_data.json"),
    indirect=True
)
async def test_response_relevancy(llm_wrapper, get_data, request):
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    response_relevancy = ResponseRelevancy(llm=llm_wrapper, embeddings=evaluator_embeddings)
    score = await response_relevancy.single_turn_ascore(get_data["sample"])

    print(f"\nPrompt: {get_data['question']}")
    print(f"Context: {get_data['contexts']}")
    print(f"Message: {get_data['message']}")
    print(f"Score: {score}")

    # Attach to HTML report
    attach_to_html_report(get_data['question'], get_data['contexts'], get_data['message'], score, request)

    assert score > 0.85
