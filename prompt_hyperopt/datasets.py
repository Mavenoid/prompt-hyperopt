import datasets
import ConfigSpace
import logging
import scipy.optimize
import numpy as np
import sys
from typing import Any, Dict, Optional
from .templatedprompt import TemplatedPrompt


def evaluate_boolean_dataset(
    trompt: TemplatedPrompt,
    engine: str,
    config: ConfigSpace.ConfigurationSpace,
    dataset=datasets.Dataset,
    dataset_context_field:Optional[str]=None, # @TODO make functions?
    dataset_question_field="question",
    dataset_answer_field="answer",
    dataset_answer_mapping={True:"{{answer_yes}}", False:"{{answer_no}}"},
    domain_context=None,
    start_index=0, #@TODO support None
    stop_index=8, #@TODO support None
    optimize_parameters:bool=False,
    optimization_loss_name="sqcost",
    answer2bias:Optional[Dict[Any,float]]=None,
    temperature:Optional[float]=None,
) -> Dict:
    # @TODO make part of task
    available_answers = set(dataset_answer_mapping.keys())
    used_answers = {dataset[i][dataset_answer_field] for i in range(start_index, stop_index)}
    if not used_answers.issubset(available_answers):
        raise ValueError()
    available_answers = sorted(available_answers)

    sample_dicts = []
    for i in range(start_index,stop_index):
        sample = dataset[i]
        known_values = dict(
            {k: v for k, v in sample.items() if k != dataset_answer_field},
            domain_context=domain_context,
            question_context=None if dataset_context_field is None else sample[dataset_context_field],
        )
        # @TODO support more classes
        # @TODO iterate over available_answers
        answer2logprob = {
            answer: trompt.get_answer_logprobs(
                engine,
                known_values,
                configuration=config,
                answer=dataset_answer_mapping[answer],
                answer_field=dataset_answer_field,
            )["total"]
            for answer in available_answers
        }
        answer2logprob = {k: v for k, v in answer2logprob.items()}
        sample_dicts.append(dict(
            answer2logprob=answer2logprob,
            correct=sample[dataset_answer_field],
        ))

    # @TODO make bias map all answers
    def get_loss(temperature, *biases, loss_name="sqcost"):
        answer2bias = dict(zip(available_answers, list(biases) + [0.]))
        totloss = 0
        for sd in sample_dicts:
            answer2logprob = dict(sd["answer2logprob"])
            answer2logprob = {k: (answer2bias[k] + v)/temperature for k, v in answer2logprob.items()}
            answer2logprob = {k: np.clip(v, np.log(sys.float_info.min*len(answer2logprob))/2, -np.log(sys.float_info.min*len(answer2logprob))/2)  for k, v in answer2logprob.items()}
            norm_factor = np.log(sum(np.exp(x) for x in answer2logprob.values()))
            answer2logprob = {k: v - norm_factor for k, v in answer2logprob.items()}
            if loss_name == "sqcost":
                totloss += (1-np.exp(answer2logprob[sd["correct"]]))**2
            elif loss_name == "logloss":
                totloss -= answer2logprob[sd["correct"]]
            elif loss_name == "accuracy":
                totloss += sd["correct"] == sorted(answer2logprob.items(), key=lambda x: -x[1])[0][0]
            else:
                raise ValueError()
        return totloss / len(sample_dicts)
    
    if optimize_parameters:
        opt = scipy.optimize.minimize(
            lambda x: get_loss(*x, loss_name=optimization_loss_name),
            x0=[1] + [0] * (len(available_answers)-1),
            bounds=[(1e-3,1e3)] + [(-99,99)] * (len(available_answers)-1),
        )
        temperature = opt.x[0]
#         biases = {"{{answer_yes}}": opt.x[0], "{{answer_no}}": 0}
        answer2bias = dict(zip(available_answers, list(opt.x[1:]) + [0.]))
    else:
        temperature = 1 if temperature is None else temperature
#         biases = {"{{answer_yes}}": 0, "{{answer_no}}": 0} if biases is None else biases
        if answer2bias is None:
            answer2bias = dict(zip(available_answers, [0.] * len(available_answers)))

    biases = list(answer2bias.values())
    biases = [x-biases[-1] for x in biases[:-1]]
    for i, bias in enumerate(biases):
        if bias == 0.:
            logging.error(
                "Optimized bias is zero - this is unexpected other than for trivial datasets."
            )
            biases[i] = 1e-3
    x = get_loss(temperature, *biases, loss_name="logloss")
    return dict(
        accuracy=get_loss(temperature, *biases, loss_name="accuracy"),
        logloss=get_loss(temperature, *biases, loss_name="logloss"),
        sqcost=get_loss(temperature, *biases, loss_name="sqcost"),
        answer2bias=answer2bias,
        temperature=temperature,
    )
