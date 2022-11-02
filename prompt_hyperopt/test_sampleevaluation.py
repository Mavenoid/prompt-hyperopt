import pytest

from prompt_hyperopt.templatedprompt import TemplatedPrompt
from prompt_hyperopt import sampleevaluation


@pytest.fixture
def singular_true_false_trompt():
    return TemplatedPrompt(
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

def test_get_samples_answers_loss(true_false_samples):
    accuracy_loss = sampleevaluation.get_samples_answers_loss(
        available_answers=["True", "False"],
        samples=true_false_samples + true_false_samples,
        sample_answer_logprobs=[
            {"True":-1, "False":-2},
            {"True":-1, "False":-3},
            {"True":-3, "False":-2},
            {"True":-2, "False":-3},
        ],
        temperature=1,
        biases=[0,0],
        dataset_answer_field="truthfulness",
        loss_name="accuracy",
    )
    assert accuracy_loss == 0.25
