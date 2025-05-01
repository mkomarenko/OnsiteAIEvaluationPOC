import pytest
from ragas.metrics import LLMContextPrecisionWithoutReference

from utils import load_test_data, attach_to_html_report


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_data",
    load_test_data("Test_data.json"),
    indirect=True
)
async def test_context_precision(llm_wrapper, get_data, request):
    context_precision = LLMContextPrecisionWithoutReference(llm=llm_wrapper)
    score = await context_precision.single_turn_ascore(get_data["sample"])

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

    assert score > 0.85
