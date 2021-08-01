from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

import ConfigSpace
import datasets
import jinja2, jinja2.nativetypes
import numpy as np

import configurationspace
import optimization
import gpt


@dataclass
class TemplatedPrompt:
    """Templated prompt which can be optimized."""

    prompt: str
    available_answers: Optional[List[str]] # @TODO should this be here?
    options: Dict[str,Tuple[Any,List[Any]]]

    # Optimized parameters.
    _configuration_space: ConfigSpace.ConfigurationSpace = field(compare=False, default=None, hash=False, init=False, repr=False)
    _configuration: ConfigSpace.Configuration = field(init=False, default=None)

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

    def _format_options(
        self,
        configuration: Optional[ConfigSpace.Configuration]=None,
        **known_values: Dict,
    ) -> Dict[str,Any]:
        # @TODO optimize
        configuration = configuration or self._configuration

        formatted = {}
        for k, v in configuration.get_dictionary().items():
            key = k.split("____")[-1]
            if isinstance(v, int):
                available_values = configuration.configuration_space.get_hyperparameters_dict()[k].meta.get("values")
                if available_values:
                    formatted[key] = available_values[v]
                    continue
            formatted[key] = v

        formatted.update(known_values)
        env = jinja2.nativetypes.NativeEnvironment(
            keep_trailing_newline=True,
            optimized = False,
            extensions=["jinja2.ext.debug","jinja2.ext.do","jinja2.ext.loopcontrols","jinja2.ext.with_"],
        )

        while True:
            any_change = False
            last_exception = None
            for k, v in formatted.items():
                if isinstance(v, str):
                    try:
                        args = dict(globals(), **formatted) # @TODO locals or not
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
        configuration: Optional[ConfigSpace.Configuration]=None,
        **known_values: Dict,
    ) -> str:
        return self._format_options(
            configuration=configuration,
            **known_values,
        )["prompt"].strip()

    # @TODO clean up
    def get_answer_logprobs(
        self,
        engine: str,
        known_values:Dict,
        answer:str,
        configuration: Optional[ConfigSpace.Configuration]=None,
        answer_field:str="answer",
    ) -> Dict:
        """Returns a dict with token_logprobs, mean, and total."""
        configuration = configuration or self._configuration
        empty_prompt = self(
            configuration,
            **{answer_field: gpt.ENDOFANSWER, **known_values},
        )
        reference_prompt = self(
            configuration,
            **{answer_field: answer+gpt.ENDOFANSWER, **known_values},
        )
        evaluation_prompt = self(
            configuration,
            **{answer_field: answer, **known_values},
        )
        if gpt.ENDOFANSWER not in empty_prompt or gpt.ENDOFANSWER not in reference_prompt:
            raise ValueError("Start-of-answer token must be encoded exactly.")

        completion_start_index = empty_prompt.rfind(gpt.ENDOFANSWER)
        completion_start_token_position = len(gpt.gpt_tokenizer.encode(empty_prompt[:completion_start_index].strip()))
        completion_end_index = reference_prompt.rfind(gpt.ENDOFANSWER)
        completion_end_token_position = len(gpt.gpt_tokenizer.encode(reference_prompt[:completion_end_index].strip()))

        if completion_start_token_position == completion_end_token_position:
            raise ValueError()

        logprobs = gpt.get_model_logprobs(engine=engine, prompt=evaluation_prompt.strip())
        answer_logprobs = logprobs[completion_start_token_position:completion_end_token_position]

        if not len(answer_logprobs):
            raise ValueError()

        return dict(
            token_logprobs=answer_logprobs,
            mean=np.mean(answer_logprobs),
            total=np.sum(answer_logprobs),
        )

    # @TODO move evaluate_task
    def optimize_greedily(self, engine:str, evaluate_task, examples) -> None:
        self._configuration = optimization.configuration_space_greedy_climb(
            self._configuration_space,
            lambda config: evaluate_task(
                engine,
                config,
                optimize_parameters=True,
                optimization_loss_name="sqcost",
                start_index=0,
                stop_index=len(examples),
                dataset=examples,
                dataset_context_field=None,
                dataset_question_field="sentence",
                dataset_answer_field="sentiment",
                dataset_answer_mapping={"Positive":"{{answer_positive}}", "Negative":"{{answer_negative}}", "Neutral":"{{answer_neutral}}"},
            ),
            lambda results: results["sqcost"] * (1 + results["logloss"]),
            max_iterations=32,
        )
