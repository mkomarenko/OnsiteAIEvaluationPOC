import pytest
from ragas import EvaluationDataset, evaluate
from ragas.metrics import ResponseRelevancy, Faithfulness, LLMContextPrecisionWithoutReference, LLMContextRecall, \
    FactualCorrectness, RubricsScore

from utils import load_test_data, attach_to_html_report


# @pytest.mark.skip(reason="This test is currently under development")
@pytest.mark.parametrize("get_data",
                         load_test_data("Test_data.json"), indirect=True)
@pytest.mark.asyncio
async def test_all_metrics(llm_wrapper, get_data, request):
    rubrics = {
        "score1_description": "The response is incorrect, irrelevant, or does not align with the ground truth.",
        "score2_description": "The response partially matches the ground truth but includes significant errors, omissions, or irrelevant information.",
        "score3_description": "The response generally aligns with the ground truth but may lack detail, clarity, or have minor inaccuracies.",
        "score4_description": "The response is mostly accurate and aligns well with the ground truth, with only minor issues or missing details.",
        "score5_description": "The response is fully accurate, aligns completely with the ground truth, and is clear and detailed.",
    }

    metrics = [LLMContextPrecisionWithoutReference(llm=llm_wrapper),
               LLMContextRecall(llm=llm_wrapper),
               Faithfulness(llm=llm_wrapper),
               FactualCorrectness(llm=llm_wrapper),
               ResponseRelevancy(llm=llm_wrapper),
               # RubricsScore(rubrics=rubrics, llm=llm_wrapper)
               ]

    eval_dataset = EvaluationDataset([get_data["sample"]])
    results = evaluate(dataset=eval_dataset, metrics=metrics)
    print('--' * 50)
    print(f"\nPrompt: {get_data['question']}")
    print(f"Context: {get_data['contexts']}")
    print(f"Message: {get_data['message']}")
    print(f"Reference: {get_data['reference']}")
    print(results)
    results.upload()

    context_precision_score = results['llm_context_precision_without_reference'][0]
    context_recall_score = results['context_recall'][0]
    faithfulness_score = results['faithfulness'][0]
    factual_correctness_score = results['factual_correctness(mode=f1)'][0]
    answer_relevancy_score = results['answer_relevancy'][0]
    # domain_specific_rubrics_score = results['domain_specific_rubrics'][0]

    errors = []
    if not context_precision_score > 0.85:
        errors.append(f"llm_context_precision_without_reference actual: {context_precision_score}, expected: > 0.85")
    if not context_recall_score > 0.8:
        errors.append(f"context_recall actual: {context_recall_score}, expected: > 0.85")
    if not faithfulness_score > 0.6:
        errors.append(f"faithfulness actual: {faithfulness_score}, expected: > 0.85")
    if not factual_correctness_score > 0.5:
        errors.append(f"factual_correctness(mode=f1) actual: {factual_correctness_score}, expected: > 0.75")
    if not answer_relevancy_score > 0.75:
        errors.append(f"answer_relevancy actual: {answer_relevancy_score}, expected: > 0.85")
    # if not domain_specific_rubrics_score > 4:
    #     errors.append(f"domain_specific_rubrics actual: {domain_specific_rubrics_score}, expected: > 4")

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
