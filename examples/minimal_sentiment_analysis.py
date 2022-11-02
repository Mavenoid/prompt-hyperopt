# Todo remove
import os.path
import sys

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)
# --

from prompt_hyperopt import TemplatedPrompt


import logging


logging.basicConfig(level=logging.DEBUG)

# Decrease logigng spam
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


trompt = TemplatedPrompt(
    prompt="""{{preamble}}

Statement: {{statement}}
Sentiment: {{answer}}
""",
    options=dict(
        answer_positive=["Positive", "happy", "Positive sentiment", "🙂", "😀"],
        answer_negative=["Negative", "sad", "Negative sentiment", "☹", "😡", "😞"],
        answer_neutral=[
            "Neutral",
            "neither",
            "ambivalent",
            "Neutral sentiment",
            "😐",
            "😶",
        ],
        preamble=[
            # "",
            "Sentiment analysis",
            "Tell whether the sentence is Positive or Negative",
        ],
    ),
    # @TODO should we specify answers or not?
)

examples = [
    dict(sentence="I am happy.", sentiment="Positive"),
    dict(sentence="I am sad.", sentiment="Negative"),
    dict(sentence="I am", sentiment="Neutral"),
]

tropt = trompt.optimize(
    "gpt2",
    examples=examples,
    features_field_mapping={"statement": "sentence"},
    targets_field_mapping={"answer": "sentiment"},
    targets_value_mapping={
        "answer": {
            "Positive": "{{answer_positive}}",
            "Negative": "{{answer_negative}}",
            "Neutral": "{{answer_neutral}}",
        }
    },
)["optimized_prompt"]

print("Optimized prompt")
print(tropt(statement="<sentence>"))


# @TODO complete
print(repr(tropt.predict(engine="gpt2", sentence="They always put too much cream")))
