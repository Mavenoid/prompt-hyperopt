import pytest

import ConfigSpace

import templatedprompt
import sampleevaluation


@pytest.fixture
def singular_true_false_trompt():
    return templatedprompt.TemplatedPrompt(
        prompt="Statement: {{statement}}. {{question_label}}: {{truthfulness}}",
        available_answers=["True", "False"],
        options=dict(
            question_label=["True or False?"],
        ),
    )

@pytest.fixture
def true_false_samples():
    return [
        dict(statement="1+3=4", truthfulness="True"),
        dict(statement="2+2=3", truthfulness="False"),
    ]

def test_get_samples_answer_logprobs(
    singular_true_false_trompt,
    true_false_samples,
):
    sample_logprobs = sampleevaluation.get_samples_answer_logprobs(
        trompt=singular_true_false_trompt,
        configuration=singular_true_false_trompt._configuration,
        engine="distilgpt2",
        samples=true_false_samples,
        dataset_answer_field="truthfulness",
        dataset_answer_mapping={"True":"True", "False":"False"},
    )
    assert sample_logprobs[0]["True"]-sample_logprobs[0]["False"] > sample_logprobs[1]["True"]-sample_logprobs[1]["False"]
