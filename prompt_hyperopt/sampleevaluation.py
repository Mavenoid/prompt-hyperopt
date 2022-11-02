from dataclasses import dataclass
import datasets
from typing import Any, Dict, Optional, List, Union, TYPE_CHECKING

import ConfigSpace
import numpy as np
import scipy.optimize
import sys

if TYPE_CHECKING:
    from prompt_hyperopt.templatedprompt import TemplatedPrompt


@dataclass
class _RemappedSample:
    features: Dict[str, str]
    target_field_name: str
    target_dataset_field_name: str
    target_value: str
    target_available_values: List[str]
    target_available_value2label: List[str]  # @TODO restructure
    domain_context: Optional[str] = None
    sample_context: Optional[str] = None


def _remap_samples(
    samples,
    features_field_mapping: Dict[str, str],
    targets_field_mapping: Dict[str, str],
    targets_value_mapping: Optional[Dict[str, Dict[Any, str]]] = None,
    # @TODO drop in favor of features?
    domain_context: Optional[str] = None,
    context_dataset_field: Optional[str] = None,
) -> List[_RemappedSample]:
    if len(targets_field_mapping) > 1:
        raise NotImplementedError(
            "Currently only optimizing a single target field at a time is supported."
        )
    elif len(targets_field_mapping) == 0:
        raise ValueError("No target field specified.")
    answer_field = list(targets_field_mapping.keys())[0]
    answer_dataset_field = targets_field_mapping[answer_field]

    used_dataset_answers = {sample[answer_dataset_field] for sample in samples}
    # @TODO rename
    available_answers_and_labels = []
    if answer_field in (targets_value_mapping or {}):
        for label, answer in targets_value_mapping[answer_field].items():
            available_answers_and_labels.append((answer, label))
        for label in used_dataset_answers:
            if not any(label == label for _, label in available_answers_and_labels):
                raise ValueError(
                    f"Label {label} in dataset is not present in targets_value_mapping."
                )
    else:
        for answer in used_dataset_answers:
            available_answers_and_labels.append((answer, answer))

    new_samples = []
    for sample in samples:
        new_sample = _RemappedSample(
            # @TODO include everything by default
            features={k: sample[v] for k, v in features_field_mapping.items()},
            target_field_name=answer_field,
            target_value=((targets_value_mapping or {}).get(answer_field, {})).get(
                sample[answer_dataset_field], sample[answer_dataset_field]
            ),
            target_available_values=[x[0] for x in available_answers_and_labels],
            target_available_value2label=dict(available_answers_and_labels),
            target_dataset_field_name=answer_dataset_field,
            domain_context=domain_context,
            sample_context=None
            if context_dataset_field is None
            else sample[context_dataset_field],
        )
        new_samples.append(new_sample)
    return new_samples


def _get_samples_answer_logprobs(
    trompt: "TemplatedPrompt",
    configuration: ConfigSpace.Configuration,
    engine: str,
    samples: List[_RemappedSample],
) -> List[Dict]:
    sample_dicts = []
    for sample in samples:
        known_values = dict(
            sample.features,
            # @TODO merge into features?
            domain_context=sample.domain_context,
            # @TODO rename from question_context?
            question_context=sample.sample_context,
        )
        answer2logprob = {
            answer: trompt._get_answer_logprobs(
                engine=engine,
                known_values=known_values,
                answer=answer,
                configuration=configuration,
                answer_field=sample.target_field_name,
            )["total"]
            for answer in sample.target_available_values
        }
        answer2logprob = {k: v for k, v in answer2logprob.items()}
        sample_dicts.append(answer2logprob)
    return sample_dicts


def _get_samples_answers_loss(
    samples: List[_RemappedSample],
    sample_answer_logprobs: List[Dict],
    temperature: float,
    biases: List[float],
    loss_name: str = "sqcost",
) -> float:
    if loss_name not in ["sqcost", "logloss", "accuracy"]:
        raise NotImplementedError()
    if len(samples) != len(sample_answer_logprobs):
        raise ValueError()

    available_answers = samples[0].target_available_values
    answer2bias = dict(zip(available_answers, list(biases) + [0.0]))
    totloss = 0
    for sample, answer_logprobs in zip(samples, sample_answer_logprobs):
        correct_answer = sample.target_value
        answer2logprob = dict(answer_logprobs)
        answer2logprob = {
            k: (answer2bias[k] + v) / temperature for k, v in answer2logprob.items()
        }
        answer2logprob = {
            k: np.clip(
                v,
                np.log(sys.float_info.min * len(answer2logprob)) / 2,
                -np.log(sys.float_info.min * len(answer2logprob)) / 2,
            )
            for k, v in answer2logprob.items()
        }
        norm_factor = np.log(sum(np.exp(x) for x in answer2logprob.values()))
        answer2logprob = {k: v - norm_factor for k, v in answer2logprob.items()}
        if loss_name == "sqcost":
            totloss += (1 - np.exp(answer2logprob[correct_answer])) ** 2
        elif loss_name == "logloss":
            totloss -= answer2logprob[correct_answer]
        elif loss_name == "accuracy":
            totloss += (
                correct_answer
                == sorted(answer2logprob.items(), key=lambda x: -x[1])[0][0]
            )
        else:
            raise ValueError()
    return totloss / len(samples)


def _optimize_samples_answers_parameters(
    samples: List[_RemappedSample],
    sample_answer_logprobs: List[Dict],
    optimization_loss_name="sqcost",
) -> Dict:
    available_answers = list(samples[0].target_available_values)
    opt = scipy.optimize.minimize(
        lambda x: _get_samples_answers_loss(
            samples=samples,
            sample_answer_logprobs=sample_answer_logprobs,
            temperature=x[0],
            biases=x[1:],
            loss_name=optimization_loss_name,
        ),
        x0=[1] + [0] * (len(available_answers) - 1),
        bounds=[(1e-3, 1e3)] + [(-99, 99)] * (len(available_answers) - 1),
    )
    temperature = opt.x[0]
    answer2bias = dict(zip(available_answers, list(opt.x[1:]) + [0.0]))
    return dict(
        temperature=temperature,
        answer2bias=answer2bias,
    )


def _evaluate_remapped_samples_answers(
    samples: List[Dict],
    sample_answer_logprobs: List[Dict],
    answer2bias: Optional[Dict[Any, float]] = None,
    temperature: Optional[float] = None,
) -> Dict:
    temperature = 1 if temperature is None else temperature
    available_answers = samples[0].target_available_values
    if answer2bias is None:
        answer2bias = dict(zip(available_answers, [0.0] * len(available_answers)))
    biases = list(answer2bias.values())
    biases = [x - biases[-1] for x in biases[:-1]]

    def get_loss(loss_name):
        return _get_samples_answers_loss(
            samples=samples,
            sample_answer_logprobs=sample_answer_logprobs,
            temperature=temperature,
            biases=biases,
            loss_name=loss_name,
        )

    return dict(
        accuracy=get_loss("accuracy"),
        logloss=get_loss("logloss"),
        sqcost=get_loss("sqcost"),
        answer2bias=answer2bias,
        temperature=temperature,
    )


def evaluate_samples_answers(
    samples: List[Dict],
    sample_answer_logprobs: List[Dict],
    # @TODO update optionals
    answer2bias: Optional[Dict[Any, float]] = None,
    temperature: Optional[float] = None,
    features_field_mapping: Optional[Dict[str, str]] = None,
    targets_field_mapping: Optional[Dict[str, str]] = None,
    targets_value_mapping: Optional[Dict[str, Dict[str, str]]] = None,
    domain_context: Optional[str] = None,
    context_dataset_field: Optional[str] = None,
) -> Dict:
    remapped_samples = _remap_samples(
        samples=samples,
        features_field_mapping=features_field_mapping,
        targets_field_mapping=targets_field_mapping,
        targets_value_mapping=targets_value_mapping,
        domain_context=domain_context,
        context_dataset_field=context_dataset_field,
    )
    return _evaluate_remapped_samples_answers(
        samples=remapped_samples,
        sample_answer_logprobs=sample_answer_logprobs,
        answer2bias=answer2bias,
        temperature=temperature,
    )


def optimize_and_evaluate_trompt_samples(
    trompt: "TemplatedPrompt",
    engine: str,
    configuration: ConfigSpace.Configuration,
    samples: Union[List[Dict], "datasets.Dataset"],
    # dataset_answer_field:str="answer",
    # dataset_answer_mapping:Dict[Any,str]={True:"{{answer_yes}}", False:"{{answer_no}}"},
    # dataset_context_field:Optional[str]=None, # @TODO make functions?
    features_field_mapping: Dict[str, str],
    targets_field_mapping: Dict[str, str],
    targets_value_mapping: Dict[str, Dict[Any, str]],
    # @TODO used how?
    domain_context: Optional[str] = None,
    context_dataset_field: Optional[str] = None,
    optimization_loss_name: str = "sqcost",
) -> Dict:
    remapped_samples = _remap_samples(
        samples=samples,
        features_field_mapping=features_field_mapping,
        targets_field_mapping=targets_field_mapping,
        targets_value_mapping=targets_value_mapping,
        domain_context=domain_context,
        context_dataset_field=context_dataset_field,
    )

    sample_answer_logprobs = _get_samples_answer_logprobs(
        trompt=trompt,
        configuration=configuration,
        engine=engine,
        samples=remapped_samples,
    )
    optimal_parameters = _optimize_samples_answers_parameters(
        samples=remapped_samples,
        sample_answer_logprobs=sample_answer_logprobs,
        optimization_loss_name=optimization_loss_name,
    )
    return _evaluate_remapped_samples_answers(
        samples=remapped_samples,
        sample_answer_logprobs=sample_answer_logprobs,
        temperature=optimal_parameters["temperature"],
        answer2bias=optimal_parameters["answer2bias"],
    )
