import functools
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

import ConfigSpace
import jinja2, jinja2.nativetypes
import numpy as np

from prompt_hyperopt import configurationspace
from prompt_hyperopt import optimization
from prompt_hyperopt import gpt
from prompt_hyperopt import sampleevaluation


@dataclass
class TemplatedPrompt:
    """Templated prompt which can be optimized.

    The templated prompt consists of a prompt string in the form of a
    jinja2 template, and a number of lists of value options for
    variables in the template. The prompt is formatted by replacing
    variables with values from the options lists, along with any
    provided known values. The prompt is then evaluated by the
    provided engine, and the result is used to optimize the
    configuration of the prompt.

    Parameters
    ----------
    prompt : str
        The prompt string, in the form of a jinja2 template.
    options : Dict[str, List[Any]]
        A dictionary of lists of options for variables in the prompt.
        These too can be jinja2 templates and options can be filled
        recursively.
    """

    prompt: str
    options: Dict[str, Tuple[Any, List[Any]]]

    # Optimized parameters.
    _configuration_space: ConfigSpace.ConfigurationSpace = field(
        compare=False, default=None, hash=False, init=False, repr=False
    )
    _configuration: ConfigSpace.Configuration = field(init=False, default=None)
    _temperature: float = field(init=False, default=1.0)
    _answer2bias: Dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._configuration_space = self._make_configuration_space()
        self._configuration = self._configuration_space.sample_configuration()

    def _make_configuration_space(self) -> ConfigSpace.ConfigurationSpace:
        return configurationspace.make_configuration_space_from_option_dict(
            dict(
                self.options,
                prompt=self.prompt,
            )
        )

    def get_configuration_space(self):
        return self._configuration_space

    def _format_options(
        self,
        configuration: Optional[ConfigSpace.Configuration] = None,
        **known_values: Dict,
    ) -> Dict[str, Any]:
        # @TODO optimize
        configuration = configuration or self._configuration

        formatted = {}
        for k, v in configuration.get_dictionary().items():
            key = k.split("____")[-1]
            if isinstance(v, int):
                # @TODO change
                available_values = (
                    configuration.configuration_space.get_hyperparameters_dict()[
                        k
                    ].meta.get("values")
                )
                if available_values:
                    formatted[key] = available_values[v]
                    continue
            formatted[key] = v
        formatted.update(known_values)

        env = jinja2.nativetypes.NativeEnvironment(
            keep_trailing_newline=True,
            optimized=False,
            extensions=["jinja2.ext.debug", "jinja2.ext.do", "jinja2.ext.loopcontrols"],
        )
        while True:
            any_change = False
            last_exception = None
            for k, v in formatted.items():
                if isinstance(v, str):
                    try:
                        args = dict(globals(), **formatted)  # @TODO locals or not
                        nv = env.from_string(v).render(**args)
                        if nv != v:
                            formatted[k] = nv
                            any_change = True
                    except NameError as e:
                        last_exception = e
                        pass
                    except jinja2.exceptions.UndefinedError as e:
                        last_exception = e
                        pass
            if not any_change:
                if last_exception:
                    raise last_exception
                break
        return formatted

    def __call__(
        self,
        configuration: Optional[ConfigSpace.Configuration] = None,
        **known_values: Dict,
    ) -> str:
        """Format prompt with the current configuration and provided arguments"""
        return self._format_options(
            configuration=configuration,
            **known_values,
        )["prompt"].strip()

    # @TODO clean up
    def _get_answer_logprobs(
        self,
        engine: str,
        known_values: Dict,
        answer: str,
        configuration: Optional[ConfigSpace.Configuration] = None,
        answer_field: str = "answer",
    ) -> Dict:
        """Returns a dict with token_logprobs, mean, and total."""
        configuration = configuration or self._configuration
        empty_prompt = self(
            configuration,
            **{answer_field: gpt.ENDOFANSWER, **known_values},
        )
        reference_prompt = self(
            configuration,
            **{answer_field: answer + gpt.ENDOFANSWER, **known_values},
        )
        evaluation_prompt = self(
            configuration,
            **{answer_field: answer, **known_values},
        )
        if (
            gpt.ENDOFANSWER not in empty_prompt
            or gpt.ENDOFANSWER not in reference_prompt
        ):
            raise ValueError("Start-of-answer token must be encoded exactly.")

        completion_start_index = empty_prompt.rfind(gpt.ENDOFANSWER)
        completion_start_token_position = len(
            gpt.gpt_tokenizer.encode(empty_prompt[:completion_start_index].strip())
        )
        completion_end_index = reference_prompt.rfind(gpt.ENDOFANSWER)
        completion_end_token_position = len(
            gpt.gpt_tokenizer.encode(reference_prompt[:completion_end_index].strip())
        )

        if completion_start_token_position == completion_end_token_position:
            raise ValueError()

        logprobs = gpt.get_model_logprobs(
            engine=engine, prompt=evaluation_prompt.strip()
        )
        answer_logprobs = logprobs[
            completion_start_token_position:completion_end_token_position
        ]

        if not len(answer_logprobs):
            raise ValueError()

        return dict(
            token_logprobs=answer_logprobs,
            mean=np.mean(answer_logprobs),
            total=np.sum(answer_logprobs),
        )

    # @TODO move evaluate_task
    def optimize_greedily(
        self,
        engine: str,
        examples,
        # question_field:str="question",
        # answer_field:str="answer",
        # @TODO maybe reverse these?
        # @TODO rename these?
        # Template field -> Dataset field
        targets_field_mapping: Dict[str, str],
        features_field_mapping: Optional[Dict[str, str]] = None,
        # Dataset field -> Dataset value -> Template value
        targets_value_mapping: Optional[Dict[str, Dict[Any, str]]] = None,
    ) -> None:
        """
        Optimize the prompt greedily by evaluating the prompt on the provided examples.

        Parameters
        ----------
        engine : str
            The engine to use for evaluation. Currently this can either
            be the name of an OpenAI engine name, or a huggingface model
            name.
        """
        (
            best_configuration,
            best_results,
            best_cost,
        ) = optimization.configuration_space_greedy_climb(
            self._configuration_space,
            lambda config: sampleevaluation.optimize_and_evaluate_trompt_samples(
                self,
                engine=engine,
                configuration=config,  # @TODO rename
                samples=examples,
                features_field_mapping=features_field_mapping,
                targets_field_mapping=targets_field_mapping,
                targets_value_mapping=targets_value_mapping,
            ),
            lambda results: results["sqcost"] * (1 + results["logloss"]),
            max_iterations=32,
        )
        # @TODO remove
        self._configuration = best_configuration
        self._temperature = best_results["temperature"]
        self._answer2bias = best_results["answer2bias"]
        return dict(
            configuration=self._configuration,
            cost=best_cost,
            accuracy=best_results["accuracy"],
            logloss=best_results["logloss"],
            temperature=best_results["temperature"],
            answer2bias=best_results["answer2bias"],
            # @TODO avoid this repetition
            optimized_prompt=OptimizedPrompt(
                template=self,
                configuration=best_configuration,
                temperature=best_results["temperature"],
                answer2bias=best_results["answer2bias"],
            ),
        )

    @functools.wraps(optimize_greedily)
    def optimize(self, *args, **kwargs):
        return self.optimize_greedily(*args, **kwargs)


@dataclass
class OptimizedPrompt:
    """Optimized templated prompt. Can be used by filling
    remaining slots."""

    template: TemplatedPrompt
    configuration: ConfigSpace.Configuration
    temperature: float = 1.0
    answer2bias: Dict[str, float] = field(default_factory=dict)

    def __call__(self, **known_values: Dict) -> str:
        return self.template(
            configuration=self.configuration,
            **known_values,
        )

    def predict_proba(
        self,
        engine: str,
        known_values: Dict,
        answer_field: str = "answer",
    ) -> Dict:
        answer2logprob_data = {}
        for answer in self.answer2bias.keys():
            answer2logprob_data[answer] = self.template._get_answer_logprobs(
                engine=engine,
                known_values=known_values,
                answer=answer,
                configuration=self.configuration,
                answer_field=answer_field,
            )
        answer2prob = {
            answer: np.exp(
                (data["total"] + self.answer2bias[answer]) / (1e-3 + self.temperature)
            )
            for answer, data in answer2logprob_data.items()
        }
        total = sum(answer2prob.values())
        answer2prob = {
            answer: logprob / total for answer, logprob in answer2prob.items()
        }
        return answer2prob

    def predict(
        self,
        engine: str,
        **known_values: Dict,
    ) -> str:
        answer2prob = self.predict_proba(
            engine=engine,
            known_values=known_values,
        )
        return sorted(answer2prob.items(), key=lambda x: x[1], reverse=True)[0][0]
