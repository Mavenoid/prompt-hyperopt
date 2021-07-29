from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import Constant, CategoricalHyperparameter
from typing import Any, Callable, Optional, Tuple
import numpy as np
import random
import logging


logger = logging.getLogger(__name__)


def configuration_space_greedy_climb(
    configuration_space: ConfigurationSpace,
    evaluator: Callable[[Configuration], Any],
    cost_getter: Callable[[Any], float] = lambda x: x,
    initial_configuration: Optional[Configuration] = None,
    random_sampler: Optional[Callable[[], Configuration]] = None,
    max_iterations: Optional[int] = None,
    random_exploration_chance: float = 0.2,
) -> Tuple[Configuration, Any, float]:
    """
    Find a local optima in the configuration space through dimension-wise
    greedy hill climbing.

    Returns the best found configuration, its results from the evaluator,
    and its cost.
    """
    if random_sampler is None:
        random_sampler = lambda: configuration_space.sample_configuration()
    if initial_configuration is None:
        initial_configuration = random_sampler()
    for hp in configuration_space.get_hyperparameters():
        if not isinstance(hp, CategoricalHyperparameter):
            if not isinstance(hp, Constant):
                raise NotImplementedError()

    # Compact representation
    best_arr_conf = np.nan_to_num(initial_configuration.get_array())
    best_configuration = initial_configuration
    best_results = None
    best_cost = float("inf")

    # Iterate until no change or max iterations
    next_hp_index = random.randrange(len(best_arr_conf))
    next_hp_value_index = 0
    last_change_hp_index = next_hp_index
    last_change_hp_value_index = next_hp_value_index

    last_eval = False
    # @TODO consider counting evaluations for iterations instead
    for it in range(max_iterations or 9999999):
        logger.info("Iteration %i. Current best: %f", it, best_cost)
        next_best_arr_conf = None
        if it == 0:
            next_best_arr_conf = np.nan_to_num(best_configuration.get_array())
            change = None
        elif last_eval and random.random() < random_exploration_chance:
            change = None
            next_configuration = random_sampler()
            next_best_arr_conf = np.nan_to_num(next_configuration.get_array())
        else:
            last_eval = False
            change = (next_hp_index, next_hp_value_index)
            if not isinstance(
                configuration_space.get_hyperparameters()[next_hp_index], CategoricalHyperparameter
            ):
                next_hp_index += 1
                next_hp_value_index = 0
            else:
                if best_arr_conf[next_hp_index] != next_hp_value_index:
                    next_best_arr_conf = np.copy(best_arr_conf)
                    next_best_arr_conf[next_hp_index] = next_hp_value_index
                next_hp_value_index += 1
                if next_hp_value_index == len(hp.choices):
                    next_hp_index += 1
                    next_hp_value_index = 0
            if next_hp_index == len(best_arr_conf):
                next_hp_index = 0
        if next_best_arr_conf is not None:
            next_configuration = Configuration(
                configuration_space,
                vector=next_best_arr_conf,
                allow_inactive_with_values=True,
            )
            try:
                next_configuration.is_valid_configuration()
            except:
                continue

            if change is not None:
                logger.info("Evaluating change: %s -> %r", hp.name, hp.choices[change[1]])
            else:
                logger.info("Evaluating non-neighboring configuration")

            results = evaluator(next_configuration)
            cost = cost_getter(results)
            last_eval = True
            if cost < best_cost:
                best_cost = cost
                best_configuration = next_configuration
                best_results = results
                if change is not None:
                    last_change_hp_index, last_change_hp_value_index = change
                else:
                    last_change_hp_index = next_hp_index
                    last_change_hp_value_index = next_hp_value_index
                logger.info(
                    "New best cost: %f", best_cost
                )
        if (
            change is not None
            and next_hp_index == last_change_hp_index
            and next_hp_value_index == last_change_hp_value_index
        ):
            break
    return best_configuration, best_results, best_cost
