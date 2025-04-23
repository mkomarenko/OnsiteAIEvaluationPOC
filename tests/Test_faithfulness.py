import pytest
from ragas.metrics import Faithfulness

from utils import load_test_data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_data",
    load_test_data("Test_data.json"),
    indirect=True
)
async def test_faithfulness(llm_wrapper, get_data):
    faithfulness = Faithfulness(llm=llm_wrapper)
    score = await faithfulness.single_turn_ascore(get_data)
    print(score)
    assert score > 0.85
