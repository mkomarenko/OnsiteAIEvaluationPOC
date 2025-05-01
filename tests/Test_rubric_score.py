import pytest
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import ResponseRelevancy, RubricsScore

from utils import load_test_data, attach_to_html_report


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_data",
    load_test_data("Test_data.json"),
    indirect=True
)
async def test_rubric_score(llm_wrapper, get_data, request):
    rubrics = {
        "score1_description": "The response is incorrect, irrelevant, or does not align with the ground truth.",
        "score2_description": "The response partially matches the ground truth but includes significant errors, omissions, or irrelevant information.",
        "score3_description": "The response generally aligns with the ground truth but may lack detail, clarity, or have minor inaccuracies.",
        "score4_description": "The response is mostly accurate and aligns well with the ground truth, with only minor issues or missing details.",
        "score5_description": "The response is fully accurate, aligns completely with the ground truth, and is clear and detailed.",
    }

    rubrics_score = RubricsScore(rubrics=rubrics, llm=llm_wrapper)
    score = await rubrics_score.single_turn_ascore(get_data['sample'])

    print(f"\nPrompt: {get_data['question']}")
    print(f"Context: {get_data['contexts']}")
    print(f"Message: {get_data['message']}")
    print(f"Reference: {get_data['reference']}")
    print(f"Score: {score}")

    # Attach to HTML report
    attach_to_html_report(get_data['question'],
                          get_data['contexts'],
                          get_data['message'],
                          get_data['reference'],
                          score,
                          request)

    assert score > 4
