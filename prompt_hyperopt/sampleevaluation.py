from typing import Any, Dict, Optional, List

import ConfigSpace
import numpy as np
import scipy
import sys

import templatedprompt


def get_samples_answer_logprobs(
    trompt: templatedprompt.TemplatedPrompt,
    configuration: ConfigSpace.Configuration,
    engine: str,
    samples: List[Dict],
    dataset_answer_field="answer",
    dataset_answer_mapping:Dict[Any,str]={True:"{{answer_yes}}", False:"{{answer_no}}"},
    dataset_context_field:Optional[str]=None, # @TODO make functions?
    domain_context:Optional[str]=None,
) -> List[Dict]:
    # @TODO make part of task
    available_answers = set(dataset_answer_mapping.keys())
    used_answers = {sample[dataset_answer_field] for sample in samples}
    if not used_answers.issubset(available_answers):
        raise ValueError()
    available_answers = sorted(available_answers)

    sample_dicts = []
    for sample in samples:
        known_values = dict(
            {k: v for k, v in sample.items() if k != dataset_answer_field},
            domain_context=domain_context,
            question_context=None if dataset_context_field is None else sample[dataset_context_field],
        )
        answer2logprob = {
            answer: trompt.get_answer_logprobs(
                engine=engine,
                known_values=known_values,
                answer=dataset_answer_mapping[answer],
                configuration=configuration,
                answer_field=dataset_answer_field,
            )["total"]
            for answer in available_answers
        }
        answer2logprob = {k: v for k, v in answer2logprob.items()}
        sample_dicts.append(answer2logprob)
    return sample_dicts


def get_samples_answers_loss(
    available_answers, #@TODO drop
    samples: List[Dict],
    sample_answer_logprobs: List[Dict],
    temperature:float,
    biases: List[float],
    dataset_answer_field="answer",
    loss_name:str="sqcost",
) -> float:
    # @TODO rename sqcost
    if loss_name not in ["sqcost", "logloss", "accuracy"]:
        raise NotImplementedError()
    if len(samples) != len(sample_answer_logprobs):
        raise ValueError()

    answer2bias = dict(zip(available_answers, list(biases) + [0.]))
    totloss = 0
    for sample, answer_logprobs in zip(samples, sample_answer_logprobs):
        correct_answer = sample[dataset_answer_field]
        answer2logprob = dict(answer_logprobs)
        answer2logprob = {k: (answer2bias[k] + v)/temperature for k, v in answer2logprob.items()}
        answer2logprob = {
            k: np.clip(
                v,
                np.log(sys.float_info.min*len(answer2logprob))/2,
                -np.log(sys.float_info.min*len(answer2logprob))/2
            )
            for k, v in answer2logprob.items()
        }
        norm_factor = np.log(sum(np.exp(x) for x in answer2logprob.values()))
        answer2logprob = {k: v - norm_factor for k, v in answer2logprob.items()}
        if loss_name == "sqcost":
            totloss += (1-np.exp(answer2logprob[correct_answer]))**2
        elif loss_name == "logloss":
            totloss -= answer2logprob[correct_answer]
        elif loss_name == "accuracy":
            totloss += correct_answer == sorted(answer2logprob.items(), key=lambda x: -x[1])[0][0]
        else:
            raise ValueError()
    return totloss / len(samples)


def optimize_samples_answers_parameters(
    available_answers: List[str], #@TODO drop
    samples: List[Dict],
    sample_answer_logprobs: List[Dict],
    dataset_answer_field="answer",
    optimization_loss_name="sqcost",
) -> Dict:
    opt = scipy.optimize.minimize(
        lambda x: get_samples_answers_loss(
            available_answers=available_answers,
            samples=samples,
            sample_answer_logprobs=sample_answer_logprobs,
            dataset_answer_field=dataset_answer_field,
            temperature=x[0],
            biases=x[1:],
            loss_name=optimization_loss_name
        ),
        x0=[1] + [0] * (len(available_answers)-1),
        bounds=[(1e-3,1e3)] + [(-99,99)] * (len(available_answers)-1),
    )
    temperature = opt.x[0]
    answer2bias = dict(zip(available_answers, list(opt.x[1:]) + [0.]))
    return dict(
        temperature=temperature,
        answer2bias=answer2bias,
    )


def evaluate_samples_answers(
    available_answers: List[str], #@TODO drop
    samples: List[Dict],
    sample_answer_logprobs: List[Dict],
    dataset_answer_field="answer",
    answer2bias:Optional[Dict[Any,float]]=None,
    temperature:Optional[float]=None,
) -> Dict:
    temperature = 1 if temperature is None else temperature
    if answer2bias is None:
        answer2bias = dict(zip(available_answers, [0.] * len(available_answers)))
    biases = list(answer2bias.values())
    biases = [x-biases[-1] for x in biases[:-1]]

    def get_loss(loss_name):
        return get_samples_answers_loss(
            available_answers=available_answers,
            samples=samples,
            sample_answer_logprobs=sample_answer_logprobs,
            dataset_answer_field=dataset_answer_field,
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


def optimize_and_evaluate_trompt_samples(
    trompt: templatedprompt.TemplatedPrompt,
    configuration: ConfigSpace.Configuration,
    engine: str,
    samples: List[Dict],
    dataset_answer_field:str="answer",
    dataset_answer_mapping:Dict[Any,str]={True:"{{answer_yes}}", False:"{{answer_no}}"},
    dataset_context_field:Optional[str]=None, # @TODO make functions?
    domain_context:Optional[str]=None,
    optimization_loss_name:str="sqcost",
) -> Dict:
    sample_answer_logprobs = get_samples_answer_logprobs(
        trompt=trompt,
        configuration=configuration,
        engine=engine,
        samples=samples,
        dataset_answer_field=dataset_answer_field,
        dataset_answer_mapping=dataset_answer_mapping,
        dataset_context_field=dataset_context_field,
        domain_context=domain_context,
    )
    optimal_parameters = optimize_samples_answers_parameters(
        available_answers=list(dataset_answer_mapping.keys()),
        samples=samples,
        sample_answer_logprobs=sample_answer_logprobs,
        dataset_answer_field=dataset_answer_field,
        optimization_loss_name=optimization_loss_name,
    )
    return evaluate_samples_answers(
        available_answers=list(dataset_answer_mapping.keys()),
        samples=samples,
        sample_answer_logprobs=sample_answer_logprobs,
        dataset_answer_field=dataset_answer_field,
        temperature=optimal_parameters["temperature"],
        answer2bias=optimal_parameters["answer2bias"],
    )
