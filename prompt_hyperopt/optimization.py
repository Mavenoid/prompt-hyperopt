from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import Constant, CategoricalHyperparameter
from typing import Any, Callable, Optional, Set, Tuple
import numpy as np
import random
import logging


def configuration_space_greedy_climb(
    configuration_space: ConfigurationSpace,
    evaluator: Callable[[Configuration], Any],
    cost_getter: Callable[[Any], float] = lambda x: x,
    initial_configuration: Optional[Configuration] = None,
    random_sampler: Optional[Callable[[], Configuration]] = None,
    max_iterations: Optional[int] = None,
    random_exploration_chance: float = 0.2,
    min_relative_improvement: float = 1e-3,
    warmup_iterations: Optional[int] = None,
    new_best_callback: Optional[Callable[[Configuration, Any, float], bool]] = None,
    included_hyperparameter_names: Optional[Set[str]] = None,
    excluded_hyperparameter_names: Optional[Set[str]] = None,
    early_termination_cost: Optional[float] = None,
    verbosity: int = 0,
) -> Tuple[Configuration, Any, float]:
    """
    Find a local optima in the configuration space through dimension-wise
    greedy hill climbing.

    Returns the best found configuration, its results from the evaluator,
    and its cost.

    Use the following for verbosity:

        import logging
        logging.getLogger("prompt_hyperopt.greedy").setLevel(logging.INFO)

    Note that this has to be set before running the method for the first time.
    """
    if random_sampler is None:
        random_sampler = lambda: configuration_space.sample_configuration()

    for hp in configuration_space.get_hyperparameters():
        if not isinstance(hp, CategoricalHyperparameter):
            if not isinstance(hp, Constant):
                raise NotImplementedError()

    logger = logging.getLogger("prompt_hyperopt.greedy")
    logger.setLevel(logging.INFO if verbosity > 0 else logging.WARNING)

    # Compact representation
    if initial_configuration is None:
        initial_configuration = random_sampler()
        best_results = None
        best_cost = float("inf")
    else:
        best_results = evaluator(initial_configuration)
        best_cost = cost_getter(best_results)
    best_arr_conf = np.nan_to_num(initial_configuration.get_array())
    best_configuration = initial_configuration

    # Iterate until no change or max iterations
    next_hp_index = random.randrange(len(best_arr_conf))
    next_hp_value_index = 0
    last_change_hp_index = next_hp_index
    last_change_hp_value_index = next_hp_value_index
    last_change_cost = best_cost

    last_eval = False
    # @TODO consider counting evaluations for iterations instead
    for it in range(max_iterations or 9999999):
        if verbosity >= 2:
            logger.info("Iteration %i. Current best: %f.", it, best_cost)
        next_arr_conf = None
        if it == 0:
            next_arr_conf = np.nan_to_num(best_configuration.get_array())
            change = None
        elif (last_eval and random.random() < random_exploration_chance) or it <= (
            warmup_iterations or 0
        ):
            change = None
            next_configuration = random_sampler()
            next_arr_conf = np.nan_to_num(next_configuration.get_array())
        else:
            last_eval = False
            change = (next_hp_index, next_hp_value_index)
            hp = configuration_space.get_hyperparameters()[next_hp_index]
            if (
                not isinstance(hp, CategoricalHyperparameter)
                or (
                    excluded_hyperparameter_names
                    and hp.name in excluded_hyperparameter_names
                )
                or (
                    included_hyperparameter_names
                    and hp.name not in included_hyperparameter_names
                )
            ):
                next_hp_index += 1
                next_hp_value_index = 0
            else:
                if best_arr_conf[next_hp_index] != next_hp_value_index:
                    next_arr_conf = np.copy(best_arr_conf)
                    next_arr_conf[next_hp_index] = next_hp_value_index
                next_hp_value_index += 1
                if next_hp_value_index == len(hp.choices):
                    next_hp_index += 1
                    next_hp_value_index = 0
            if next_hp_index == len(best_arr_conf):
                next_hp_index = 0
        if next_arr_conf is not None:
            next_configuration = Configuration(
                configuration_space,
                vector=next_arr_conf,
                allow_inactive_with_values=True,
            )
            try:
                next_configuration.is_valid_configuration()
            except:
                continue

            results = evaluator(next_configuration)
            cost = cost_getter(results)

            if cost < best_cost or verbosity >= 2:
                if change is not None:
                    logger.info(
                        "%s Evaluated change (cost: %f): %s -> %r.",
                        "New best!" if cost < best_cost else "No change.",
                        cost,
                        hp.name,
                        hp.choices[change[1]],
                    )
                else:
                    logger.info(
                        "%s Evaluated non-neighboring configuration (cost: %f).",
                        "New best!" if cost < best_cost else "No change.",
                        cost,
                    )

            last_eval = True
            if cost <= best_cost:
                best_cost = cost
                best_configuration = next_configuration
                best_results = results
                best_arr_conf = next_arr_conf
                if change is not None:
                    last_change_hp_index, last_change_hp_value_index = change
                elif cost < (1 + min_relative_improvement) * last_change_cost:
                    last_change_hp_index = next_hp_index
                    last_change_hp_value_index = next_hp_value_index
                    last_change_cost = best_cost
                if new_best_callback:
                    if new_best_callback(
                        best_configuration,
                        best_results,
                        best_cost,
                    ):
                        break
        if early_termination_cost is not None and best_cost <= early_termination_cost:
            break
        if (
            change is not None
            and next_hp_index == last_change_hp_index
            and next_hp_value_index == last_change_hp_value_index
        ):
            break
    return best_configuration, best_results, best_cost
