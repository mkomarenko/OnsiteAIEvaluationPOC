import pytest
from ragas.metrics import LLMContextPrecisionWithoutReference

from utils import load_test_data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_data",
    load_test_data("Test_data.json"),
    indirect=True
)
async def test_context_precision(llm_wrapper, get_data):
    context_precision = LLMContextPrecisionWithoutReference(llm=llm_wrapper)
    score = await context_precision.single_turn_ascore(get_data)
    print(score)
    assert score > 0.85
