import pytest
from ragas import EvaluationDataset, evaluate
from ragas.metrics import ResponseRelevancy, Faithfulness, LLMContextPrecisionWithoutReference

from utils import load_test_data


@pytest.mark.skip(reason="This test is currently under development")
@pytest.mark.parametrize("get_data",
                         load_test_data("Test_data.json"), indirect=True)
@pytest.mark.asyncio
async def test_all_metrics(llm_wrapper, get_data):
    metrics = [LLMContextPrecisionWithoutReference(llm=llm_wrapper),
               Faithfulness(llm=llm_wrapper),
               ResponseRelevancy(llm=llm_wrapper)
               ]

    eval_dataset = EvaluationDataset([get_data])
    results = evaluate(dataset=eval_dataset, metrics=metrics)
    print(results)
    # results.upload()
