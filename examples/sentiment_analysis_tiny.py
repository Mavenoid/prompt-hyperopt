import prompt_hyperopt.datasets
import prompt_hyperopt.optimization
from prompt_hyperopt import TemplatedPrompt

engine = "ada"

examples=[
    dict(sentence="I am happy", sentiment="Positive"),
    dict(sentence="It started good but for most of the song, all I could hear was the bass", sentiment="Negative"),
    dict(sentence=":((((", sentiment="Negative"),
    dict(sentence="Cake for the third Friday in a row! 😢😛", sentiment="Positive"),
    dict(sentence="It won't work", sentiment="Negative"),
    dict(sentence="Turn right", sentiment="Neutral"),
    dict(sentence="Nothing special to say", sentiment="Neutral"),
]

trompt = prompt_hyperopt.TemplatedPrompt(
    prompt="""{{preamble}}

Statement: {{sentence}}
Sentiment: {{sentiment}}
""",
    available_answers=[
        "{{answer_positive}}", "{{answer_negative}}", "{{answer_neutral}}"
    ], # Optional. Do we want this to be {{answer_yes}} or yes?
    options=dict(
        answer_positive=["Positive", "happy", "Positive sentiment", "🙂", "😀"],
        answer_negative=["Negative", "sad", "Negative sentiment", "☹", "😡", "😞"],
        answer_neutral=["Neutral", "neither", "ambivalent", "Neutral sentiment", "😐", "😶"],
        preamble=["", "Sentiment analysis", "Label the example as either Positive or Negative"]
    ),
)

cs = trompt.get_configuration_space()

prompt_hyperopt.optimization.configuration_space_greedy_climb(
    cs,
    lambda config: prompt_hyperopt.datasets.evaluate_boolean_dataset(
        engine,
        trompt,
        optimize_parameters=True,
        optimization_loss_name="sqcost",
        start_index=0,
        stop_index=len(examples),
        dataset=examples,
        dataset_context_field=None,
        dataset_question_field="sentence",
        dataset_answer_field="sentiment",
        # Note: These are trompts and will be filled during optimization.
        dataset_answer_mapping={
            "Positive": "{{answer_positive}}",
            "Negative": "{{answer_negative}}",
            "Neutral": "{{answer_neutral}}"},
    ),
    lambda results: results["sqcost"] * (1 + results["logloss"]),
    max_iterations=32,
)
