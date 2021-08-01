import pytest

import templatedprompt

@pytest.fixture
def single_alternative_prompt():
    return templatedprompt.TemplatedPrompt(
        prompt="The {{best_term}} color is {{answer}}",
        available_answers=["amaranth"], # @TODO drop?
        options=dict(
            best_term=["best"]
        ),
    )

def test_format_single_alternative_prompt(single_alternative_prompt):
    prompt = single_alternative_prompt(answer="amaranth")
    assert prompt == "The best color is amaranth"
