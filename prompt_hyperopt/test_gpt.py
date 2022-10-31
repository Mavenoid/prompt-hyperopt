from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from . import gpt
from .templatedprompt import TemplatedPrompt

import pytest

@pytest.fixture
def singular_qa_trompt():
    return TemplatedPrompt(
        prompt="{{question_prefix}}{{question}}\n{{answer_prefix}}{{answer}}",
        available_answers=["Yes", "No"], # @TODO drop?
        options=dict(
            question_prefix=["Question: "],
            answer_prefix=["Answer: "],
        ),
    )


def test_distilgpt2_singular_qa_trompt(singular_qa_trompt):
    known_values = dict(question="Do there exist red apples?")
    logprobs_yes = singular_qa_trompt.get_answer_logprobs(
        engine="distilgpt2",
        answer="Yes",
        known_values=known_values
    )
    logprobs_no = singular_qa_trompt.get_answer_logprobs(
        engine="distilgpt2",
        answer="No",
        known_values=known_values,
    )
    print("Yes logprobs:", logprobs_yes)
    print("No logprobs:", logprobs_no)
    assert logprobs_yes["total"] > logprobs_no["total"]
