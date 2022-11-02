import datasets
import ConfigSpace
from typing import Any, Dict, Optional, Union
from prompt_hyperopt import sampleevaluation

from prompt_hyperopt.templatedprompt import OptimizedPrompt, TemplatedPrompt


# @TODO rename as it is more general than just boolean datasets
def evaluate_boolean_dataset(
    trompt: Union[TemplatedPrompt, OptimizedPrompt],
    engine: str,
    config: ConfigSpace.ConfigurationSpace,
    dataset: "datasets.Dataset",
    features_field_mapping: Dict[str, str],
    targets_field_mapping: Dict[str, str],
    targets_value_mapping: Optional[Dict[str, Dict[Any, str]]] = None,
    available_answers=[True, False],
    domain_context=None,
    start_index=0,  # @TODO support None
    stop_index=8,  # @TODO support None
    optimize_parameters: bool = False,
    optimization_loss_name="sqcost",
    answer2bias: Optional[Dict[Any, float]] = None,
    temperature: Optional[float] = None,
    uncalibrated_confidences_weight: float = 1e-3,
) -> Dict:
    """Evaluate a TemplatedPrompt on a boolean dataset."""
    # @TODO change to only support OptimizedPrompt?
    if isinstance(trompt, OptimizedPrompt):
        if temperature is None:
            temperature = trompt.temperature
        if answer2bias is None:
            answer2bias = trompt.answer2bias
        if config is None:
            config = trompt.config
        trompt = trompt.template
    # @TODO encapsulate in sampleevaluation instead
    remapped_samples = sampleevaluation._remap_samples(
        dataset[start_index:stop_index],
        features_field_mapping=features_field_mapping,
        targets_field_mapping=targets_field_mapping,
        targets_value_mapping=targets_value_mapping,
        domain_context=domain_context,
        # @TODO context_dataset_field
    )
    sample_answer_logprobs = sampleevaluation._get_samples_answer_logprobs(
        trompt=trompt,
        configuration=config,
        engine=engine,
        samples=remapped_samples,
    )
    if optimize_parameters:
        optimal_parameters = sampleevaluation._optimize_samples_answers_parameters(
            remapped_samples,
            sample_answer_logprobs,
            optimization_loss_name=optimization_loss_name,
        )
        temperature = optimal_parameters["temperature"]
        answer2bias = optimal_parameters["answer2bias"]
    results = sampleevaluation._evaluate_remapped_samples_answers(
        samples=remapped_samples,
        sample_answer_logprobs=sample_answer_logprobs,
        temperature=temperature,
        answer2bias=answer2bias,
    )
    return results


# @TODO update
def evaluate_boolq(
    engine: str,
    trompt: TemplatedPrompt,
    config: ConfigSpace.ConfigurationSpace,
    dataset=None,
    domain_context=None,
    start_index=0,
    stop_index=8,
    optimize_parameters: bool = False,
    optimization_loss_name="sqcost",
    answer2bias: Optional[Dict] = None,
    temperature: Optional[float] = None,
) -> Dict:
    """Evaluate a TemplatedPrompt on the BoolQ dataset."""
    if dataset is None:
        dataset = datasets.load_dataset("boolq")
    return evaluate_boolean_dataset(
        trompt=trompt,
        engine=engine,
        config=config,
        dataset=dataset,
        dataset_context_field="passage",
        dataset_question_field="question",
        dataset_answer_field="answer",
        dataset_answer_mapping={True: "{{answer_yes}}", False: "{{answer_no}}"},
        domain_context=domain_context,
        start_index=start_index,
        stop_index=stop_index,
        optimize_parameters=optimize_parameters,
        answer2bias=answer2bias,
        temperature=temperature,
    )
