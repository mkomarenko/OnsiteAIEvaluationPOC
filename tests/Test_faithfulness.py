import pytest
from ragas.metrics import Faithfulness

from utils import load_test_data, attach_to_html_report


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_data",
    load_test_data("Test_data.json"),
    indirect=True
)
async def test_faithfulness(llm_wrapper, get_data, request):
    faithfulness = Faithfulness(llm=llm_wrapper)
    score = await faithfulness.single_turn_ascore(get_data["sample"])

    print(f"\nPrompt: {get_data['question']}")
    print(f"Context: {get_data['contexts']}")
    print(f"Score: {score}")

    # Attach to HTML report
    attach_to_html_report(get_data['question'], get_data['contexts'], score, request)

    assert score > 0.85
