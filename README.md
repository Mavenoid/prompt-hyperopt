![logo](assets/logo.png)

# prompt-hyperopt

More reliable prompt crafting though templates, hyperparameter optimization from few examples, and calibration of and across language models.

![tests status](https://github.com/mavenoid/prompt-hyperopt/actions/workflows/tests.yml/badge.svg)

## Why

Prompt optimization offers a stuctured way to produce more reliable generation.

Manually crafting reliable prompts can be a difficult and time-consuming task. It can seem like one experiment succeeds only to find the next example going on a seemingly random tangent. Adapting the prompt to fix one such misbehaving example often breaks a previously-tried example. There should be a better way to develop good prompts.

## The solution

Instead of ad-hoc modifications of prompts, Humans should focus on generating ideas for the different ways a task prompt can be expressed. For example, maybe you could use emojis 🙂/😞 instead of Positive/Negative for sentiment analysis? Maybe you could put quotes around the paragraph to make it clear that it is separate from the task description? Maybe you want a preamble that explains what the task is about or what the generation should not do? When you see a failed generation, you can usually think of a few ways to adjust prompts to steer it in the right direction and so iteratively add more prompt options.

The task on the human is provide these variants, and the library is responsible for finding the alternative that performs the best with respect to the examples. The library provides intuitive curly-brace templates to express variants more compactly.

By using token probabilities and tuning parameters such as `temperature`, an informative evaluation can be made even with a few examples; and by using hyperparameter-optimization approaches, the number of evaluations to find the best prompts can be kept to a minimum. Evaluations can also be cached to prevent unnecessary reruns. Long hyperparameter runs can initially also be done on smaller (i.e. faster and cheaper) models and then adapted to and tuned further on larger models. Finally, the temperature can be chosen automatically to calibrate confidence levels.

The new workflow to engineer prompts becomes:
1. Come up with an initial prompt and a few sensible variations.
2. Come up with a few examples of expected results.
3. Run the optimization.
4. Inspect the examples that fail and add a few new examples or variations to hopefully fix them.
5. Repeat 3-4 until happy with the performance.

## Details

This library provides convenient methods to express prompt alternatives for a task and use hyperparameter-optimization techniques to find the best one. This approach has the benefit that it works without access to gradients (such as with GPT3), significant time and resource budgets for optimization, and it generalizes well also for small datasets (e.g. 3-30 examples). For projects where these limitations are not factors, one should expect to see better results fine proper fine tuning or [prompt tuning](https://arxiv.org/pdf/2104.08691.pdf).

Prompt hyperparameter optimization should perform better than traditional few-shot learning prompts and worse than proper fine tuning. Even if a project can afford to fine tune however, it may be advantageous to prototype and iterate more quickly with this library.

## Features

* Seemlessly switch between API language-models such as GPT3 and locally-run models such as GPT2, GPT-J and Flan-T5.
* Optimize `temperature` and `top_p` rather than guessing.
* Calibrate token biases to get results similar to [neutral-prompt calibration](https://arxiv.org/pdf/2102.09690.pdf).
* Optimize for prompts which generate parseable results rather than going off script.
* Find prompts that most reliably produce expected outputs.
* Initiate optimization with smaller language models and recalibrate prompts for larger models.
* Minimize unproductive evaluations using Hyperband Bayesian Optimization via [hpbandster](https://automl.github.io).

## Installation

For the full installation, simply `pip install prompt_hyperopt`. Note that this
presently also includes heavy dependencies like `torch` and `transformers`.

To execute tests, use `pip install prompt_hyperopt[dev]` or `pip install -e .[dev]` if checked out.

## Getting started

### Expressing what you want

After installing, the first step to use the library is to make a handful
examples of what you expect the prompt to generate. Each input can have
one or more input values and should have exactly one output.

For example, if you want to optimize story generation, you could write a 
few examples with an input of a preceding paragraph, and the output
should be the next generated sentence.

```python
from prompt_hyperopt import PromptHyperopt

examples = [
    {
        "story": "While out on a walk, I found a $20 bill. As I was picking it up, I saw a homeless man watching me.",
        "continuation": "I handed him the money and he smiled.",
    },
    {
        "story": "The dog was hungry.",
        "continuation": "So she ate the food.",
    },
]
```

The way this optimization works, you must also provide a number of answer
options, including examples continuations that you would not like to see.

```python

possible_continuations = [
    examples[0]["continuation"],
    examples[1]["continuation"],
    "I set fire to the bill.",
    "So she ordered a pizza.",
    "Once upon a time...",
    "\n\n",
]
```

### Defining the prompt options

Next come up with a few different ways to express the prompt. This is
done using curly-brace templates option lists. You should not provide
options for the inputs and outputs of the examples.

```python
trompt = TemplatedPrompt(
    """{preamble}

{story}{separation}{continuation}""",
    options=dict(
        preamble=[
            "Write a continuation to the following story:",
            "This is the most amazing story ever:",
            "A paragraph from a best-selling novel:",
        ],
        separation=[
            "\n\n", "\n", " ",
            "\n\nStory continuation: ",
        ],
    ),
)
```

### Finding the best prompt

With the template and examples, we can now optimize the prompt. The
`TemplatedPrompt` class provides a convenient interface to do this.
In order to do the optimization, the library needs to know the
relation between the examples and the template, which is done through
the field `targets_field_mapping`. This describes what fields in the
template should be prediceted vs filled statically. For more complex
datasets, additional fields may need to be specified.

Note also that if your template is complex, this will need to
generate a lot of examples, and depending on the engine you use, this
may both take a while and be costly.

```python

engine = "text-curie-001"

trompt.optimize(
    engine,
    examples,
    targets_field_mapping={
        "continuation": "continuation",
    },
)
```

Then print the best prompt:

```python
print(trompt(story="<story>, continuation="<continuation>"))
```


## Interesting findings

Out of the box, prompts which seem to provide mostly accurate predictions for one language model do not necessarily perform well for others, even bordering on random answers. Notably this can be observed for GPT3-curie 6.7B vs GPT3-davinci 175B. What we have found however is that by recalibrating token biases for the two models, then often high-performing prompts for one model are also high-performing for the other. This can be done by running the optimize method without options.

## Examples

### Sentiment analysis

```
from prompt_hyperopt import TemplatedPrompt

trompt = TemplatedPrompt(
    prompt="""{{preamble}}

Statement: {{sentence}}
Sentiment: {{sentiment}}
""",
    options=dict(
        answer_positive=["Positive", "happy", "Positive sentiment", "🙂", "😀"],
        answer_negative=["Negative", "sad", "Negative sentiment", "☹", "😡", "😞"],
        preamble=["Sentiment analysis.", "Predict whether the sentence is Positive or Negative"]
    ),
)

examples = [
    dict(sentence="I am happy.", sentiment="Positive"),
    dict(sentence="I am sad.", sentiment="Negative"),
    dict(sentence="I am", sentiment="Neutral"),
]

trompt.optimize(
    "gpt2",
    examples,
    features_field_mapping={"sentence": "sentence"},
    targets_field_mapping={"sentiment": "sentiment"},
    targets_value_mapping={
        "sentiment": {
            "Positive": "{{answer_positive}}",
            "Negative": "{{answer_negative}}",
        }
    }
)

print("Optimized prompt:")
print(trompt(sentence="Coffee is good"))

print("Prediction:")
print(trompt.predict(sentence="Coffee is good"))
```

## Results

On [Bool-Q](https://paperswithcode.com/sota/question-answering-on-boolq), prompt hyperoptimization produces an accuracy of 81.3 % with 32 examples for GPT-3 davinci, in contrast to previous results of GPT-3 few-shot on 32 examples. This should however be confirmed and made reproducible. Note that the `davinci` model was trained before the release of the Bool-Q dataset while the `instruct` and `text` models may have had access to it.

Best found prompt for Bool-Q (with 1-shot examples):

```
Answers are helpful and accurate.
--

--
In 1907 the Peking to Paris automobile race had inspired an even bolder test of these new machines. The following year the course would be from New York City, USA, to Paris, France with a 150-mile (240 km) ship passage from Nome, Alaska, across the Bering Strait to East Cape, Siberia, this at a time when ``the motor car is the most fragile and capricious thing on earth.''

Q: 「can you drive to paris from new york」 Yes or No?
Answer. 「No」

Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.

Q: 「do iran and afghanistan speak the same language」 Yes or No?
Answer. 「Yes」
```

### License

This package has been developed by [Mavenoid](https://www.mavenoid.com) and is released under MIT License.
