# prompt-hyperopt

Optimize prompts for language models with hyperparameter optimization with few examples and generalization across different language models.

## Why

Finding reliable prompts can be a difficult and time-consuming task. It can seem like one experiment succeeds only to find the next example going on a seemingly random tangent. Trying to fix prompts for one example may break previously-tried examples, and revalidating all examples is often not an efficient workflow.

Enter prompt hyperoptimization.

It is easier to come up with ideas for the different ways a task prompt can be expressed -- Maybe you could use emojis 🙂/😞 instead of Positive/Negative for sentiment analysis? Maybe you could put quotes around the paragraph to make it clear that it is separate from the task description? Maybe you want a preamble that explains what the task is about?

The human provides this creative constrained optimization; and the library does some of the lifting. Use templates to express your ideas for prompt variants, provide a few examples of expected results, and run an optimization to find the prompt that best fit the examples.

By using token probabilities and tuning parameters such as `temperature`, an informative evaluation can be made even with a few examples; and by using hyperparameter-optimization approaches, the number of evaluations to find the best prompts can be kept to a minimum. Evaluations can also be cached to prevent unnecessary reruns.

The new workflow to engineer prompts becomes:
1. Come up with an initial prompt and a few sensible variations.
2. Come up with a few examples of expected results.
3. Run the optimization.
4. Inspect the examples that fail and add a few new examples or variations to hopefully fix them.
5. Repeat 3-4 until happy with the performance.

## How

This library provides convenient methods to express prompt alternatives for a task and use hyperparameter-optimization techniques to find the best one. This approach has the benefit that it works without access to gradients (such as with GPT3), significant time and resource budgets for optimization, and it generalizes well also for small datasets (e.g. 3-30 examples). For projects where these limitations are not factors, one should expect to see better results fine proper fine tuning or [prompt tuning](https://arxiv.org/pdf/2104.08691.pdf).

Prompt hyperparameter optimization should perform better than traditinoal few-shot learning prompts and worse than proper fine tuning. Even if a project can afford to fine tune however, it may be advantageous to prototype and iterate more quickly with this library.

## Features

* Optimize `temperature` and `top_p` rather than guessing.
* Calibrate token biases to get results similar to [neutral-prompt calibration](https://arxiv.org/pdf/2102.09690.pdf).
* Optimize for prompts which generate parseable results.
* Find prompts that best conform to expected outputs.
* Initiate optimization with smaller language models and recalibrate prompts for larger models.
* Minimize unproductive evaluations using Hyperband Bayesian Optimization via [hpbandster](https://automl.github.io).

## Installation

TODO

```
pip install prompt_hyperopt
```

## Getting started

### Evaluating a prompt variant



### Expressing prompt variants


### Finding the best prompt variant

## Interesting findings

Curiously, out of the box, prompts which seem to provide mostly accurate predictions for one language model does not perform well for others - even bordering on random answers. Notably this can be observed for GPT3-curie 6.7B vs GPT3-davinci 175B. What we have found however is that by recalibrating token biases for the two models, then often high-performing prompts for one model are also high-performing for the other. This can be done by using the utility method TODO.

## Examples

### Sentiment analysis

```
trompt = TemplatedPrompt(
    prompt="""{{preamble}}

Statement: {{sentence}}
Sentiment: {{sentiment}}
""",
    options=dict(
        answer_positive=["Positive", "happy", "Positive sentiment", "🙂", "😀"],
        answer_negative=["Negative", "sad", "Negative sentiment", "☹", "😡", "😞"],
        answer_neutral=["Neutral", "neither", "ambivalent", "Neutral sentiment", "😐", "😶"],
        preamble=["", "Sentiment analysis", "Tell whether the sentence is Positive or Negative"]
    ),
)

trompt.optimize()

trompt.complete(sentence="They always put too much cream")
```

## Results

On [Bool-Q](https://paperswithcode.com/sota/question-answering-on-boolq), prompt hyperoptimization produces an accuracy of 81.3 % with 32 examples for GPT-3 davinci, in contrast to previous results of GPT-3 few-shot on 32 examples. TODO confirm. TODO fill in more.

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
