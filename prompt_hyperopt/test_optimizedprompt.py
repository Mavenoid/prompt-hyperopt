from prompt_hyperopt import TemplatedPrompt


def test_optimize_easy_prompt():
    trompt = TemplatedPrompt(
        prompt="""{{question}} The answer (out of Yes or No) is '{{answer}}'""",
        options=dict(
            answer_true=["Yes", "No"],
            answer_false=["Yes", "No"],
        ),
    )

    examples = [
        # dict(question="Is an elephant an animal?", answer="True"),
        # dict(question="Is the moon covered in trees?", answer="False"),
        # dict(question="Is red a color?", answer="True"),
        dict(question="Is true true?", answer="True"),
        dict(question="Is true false?", answer="False"),
        dict(question="Is false false?", answer="True"),
    ]

    results = trompt.optimize(
        "gpt2",
        examples=examples,
        features_field_mapping={"question": "question"},
        targets_field_mapping={"answer": "answer"},
        targets_value_mapping={
            "answer": {
                "True": "{{answer_true}}",
                "False": "{{answer_false}}",
            }
        },
    )

    assert results["optimized_prompt"](
        question="",
        answer="{{answer_true}}",
    ) == trompt(answer="Yes")

    assert results["optimized_prompt"](
        question="",
        answer="{{answer_false}}",
    ) == trompt(answer="No")
