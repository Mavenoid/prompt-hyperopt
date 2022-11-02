import pytest

from prompt_hyperopt.templatedprompt import TemplatedPrompt
from prompt_hyperopt import sampleevaluation


@pytest.fixture
def singular_true_false_trompt():
    return TemplatedPrompt(
        prompt="Statement: {{statement}}. {{question_label}}: {{truthfulness}}",
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


@pytest.fixture
def true_false_remapped_samples(true_false_samples):
    return sampleevaluation._remap_samples(
        samples=true_false_samples,
        # @TODO make default - just hide targets
        features_field_mapping={
            "statement": "statement",
        },
        targets_field_mapping={
            "truthfulness": "truthfulness",
        },
    )


def test_get_samples_answer_logprobs(
    singular_true_false_trompt,
    true_false_remapped_samples,
):
    sample_logprobs = sampleevaluation._get_samples_answer_logprobs(
        trompt=singular_true_false_trompt,
        configuration=singular_true_false_trompt._configuration,
        engine="distilgpt2",
        samples=true_false_remapped_samples,
    )
    assert (
        sample_logprobs[0]["True"] - sample_logprobs[0]["False"]
        > sample_logprobs[1]["True"] - sample_logprobs[1]["False"]
    )


def test_get_samples_answers_loss(true_false_remapped_samples):
    accuracy_loss = sampleevaluation._get_samples_answers_loss(
        samples=true_false_remapped_samples + true_false_remapped_samples,
        # @TODO rename to make clearer
        sample_answer_logprobs=[
            {"True": -1, "False": -2},
            {"True": -1, "False": -3},
            {"True": -3, "False": -2},
            {"True": -2, "False": -3},
        ],
        temperature=1,
        biases=[0, 0],
        loss_name="accuracy",
    )
    assert accuracy_loss == 0.25
