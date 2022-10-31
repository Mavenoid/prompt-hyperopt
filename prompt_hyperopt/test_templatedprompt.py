import pytest

from .templatedprompt import TemplatedPrompt

@pytest.fixture
def single_alternative_trompt():
    return TemplatedPrompt(
        prompt="The {{best_term}} color is {{answer}}",
        available_answers=["amaranth"], # @TODO drop?
        options=dict(
            best_term=["best"]
        ),
    )

def test_format_single_alternative_trompt(single_alternative_trompt):
    prompt = single_alternative_trompt(answer="amaranth")
    assert prompt == "The best color is amaranth"
