import prompt_hyperopt.datasets
import prompt_hyperopt.optimization
from prompt_hyperopt import TemplatedPrompt
import logging


logging.basicConfig(level=logging.DEBUG)

# Decrease logigng spam
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)



# logging.getLogger("prompt_hyperopt.greedy").setLevel(logging.DEBUG)

engines = [
    "gpt2",
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-002",
]

examples=[
    dict(sentence="I am happy", sentiment="Positive"),
    dict(sentence="Price to high for a product with problems.", sentiment="Negative"),
    dict(sentence="language for thinking.", sentiment="Neutral"),
    dict(sentence="Cake for the third Friday in a row! 😢😛", sentiment="Positive"),
    dict(sentence="Solid Keyboard. Upgrade the keys, it worth it.", sentiment="Positive"),
    dict(sentence=":((((", sentiment="Negative"),
    dict(sentence="That's just what I needed today!", sentiment="Negative"),
    dict(sentence="Turn right", sentiment="Neutral"),
    # Will be used for post-optimization evaluation
    dict(sentence="Break a leg!", sentiment="Positive"),
    dict(sentence="It started good but for most of the song, all I could hear was the bass", sentiment="Negative"),
    dict(sentence="Four is greater than three", sentiment="Neutral"),
    dict(sentence="Does this come with a charger?", sentiment="Neutral"),
    dict(sentence="Design can be better", sentiment="Negative"),
    dict(sentence="Powerful device", sentiment="Positive"),
]

dev_examples = examples[:-3]
test_examples = examples[-3:]

# @TODO investigate why "" gets encoded as None

trompt = prompt_hyperopt.TemplatedPrompt(
    prompt="""{{example}}

{{preamble}}{{options}}

Statement: {{sentence}}{{separator}}{{sentiment_label}} {{sentiment}}
""",
    available_answers=[
        "{{answer_positive}}", "{{answer_negative}}", "{{answer_neutral}}"
    ], # Optional. Do we want this to be {{answer_yes}} or yes?
    options=dict(
        answer_positive=["Positive", "happy", "Positive sentiment", "🙂", "😀"],
        answer_negative=["Negative", "sad", "Negative sentiment", "☹", "😡", "😞"],
        answer_neutral=["Neutral", "neither", "ambivalent", "Neutral sentiment", "😐", "😶"],
        preamble=[" ", "Sentiment analysis.", "Assign the sentiment of the statement."],
        options=[
            " ",
            " ({{answer_positive}}/{{answer_negative}}/{{answer_neutral}})",
            " Options: {{answer_positive}}, {{answer_negative}}, {{answer_neutral}}",
        ],
        separator=[" ", "\n", "\\n", " -- "],
        sentiment_label=[
            "Sentiment:",
            "The statement's sentiment is",
            "Q: What is the sentiment? A: The answer is",
        ],
        example=[
            "",
            """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.""",
        ]
    ),
)

cs = trompt.get_configuration_space()

best_config = None
for engine in engines:
    best_config, best_results, best_cost = prompt_hyperopt.optimization.configuration_space_greedy_climb(
        cs,
        lambda config: prompt_hyperopt.datasets.evaluate_boolean_dataset(
            trompt,
            engine,
            config,
            optimize_parameters=True,
            optimization_loss_name="sqcost",
            start_index=0,
            stop_index=len(dev_examples),
            dataset=dev_examples,
            dataset_context_field=None,
            dataset_question_field="sentence",
            dataset_answer_field="sentiment",
            # Note: These are trompts and will be filled during optimization.
            dataset_answer_mapping={
                "Positive": "{{answer_positive}}",
                "Negative": "{{answer_negative}}",
                "Neutral": "{{answer_neutral}}"
            },
        ),
        lambda results: results["sqcost"] * (1 + results["logloss"]),
        initial_configuration=best_config,
        early_termination_cost=1e-3,
        max_iterations=128,
        verbosity=1,
    )

    test_results = prompt_hyperopt.datasets.evaluate_boolean_dataset(
        trompt,
        engine,
        best_config,
        optimization_loss_name="sqcost",
        start_index=0,
        stop_index=len(test_examples),
        dataset=test_examples,
        dataset_context_field=None,
        dataset_question_field="sentence",
        dataset_answer_field="sentiment",
        dataset_answer_mapping={
            "Positive": "{{answer_positive}}",
            "Negative": "{{answer_negative}}",
            "Neutral": "{{answer_neutral}}"
        },
        temperature=best_results["temperature"],
        answer2bias=best_results["answer2bias"],
    )

    print("--- Engine: %s ---" % engine)

    print("Dev results: " + repr(best_results))
    print("Test results: " + repr(test_results))

    print("Best configuration:")
    print(best_config)

    print("Best prompt:")
    print("<<<")
    print(trompt(best_config))
    print(">>>")
