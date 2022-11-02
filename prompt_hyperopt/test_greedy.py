import pytest
from typing import Dict

import difflib
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant

from prompt_hyperopt.optimization import configuration_space_greedy_climb


@pytest.fixture
def configuration_space_square_diff():
    cs = ConfigurationSpace()
    cs.add_hyperparameters(
        [
            CategoricalHyperparameter("x", choices=[0, 3, 5, 7, 9]),
            CategoricalHyperparameter("y", choices=[2, 4, 6, 8]),
        ]
    )
    return cs


@pytest.fixture
def configuration_space_with_constants():
    cs = ConfigurationSpace()
    cs.add_hyperparameters(
        [
            Constant("determiner", value="the"),
            CategoricalHyperparameter("animal", choices=["cat", "dog"]),
            Constant("verb", value="is"),
            CategoricalHyperparameter("activity", choices=["chirping", "barking"]),
        ]
    )
    return cs


def calc_squared_diff(conf_dict: Dict[str, int]) -> Dict:
    x_squared = conf_dict["x"] ** 2
    y_squared = conf_dict["y"] ** 2
    return dict(
        x_squared=x_squared, y_squared=y_squared, diff=abs(x_squared - y_squared)
    )


def test_greedy_climb_squared_diff(configuration_space_square_diff):
    best_config, best_results, best_cost = configuration_space_greedy_climb(
        configuration_space_square_diff,
        lambda config: calc_squared_diff(config.get_dictionary()),
        lambda results: results["diff"],
        initial_configuration=Configuration(
            configuration_space_square_diff, vector=(3, 2)
        ),
    )
    print("Best configuration: ", best_config)
    print("Best results:", best_results)
    print("Best cost:", best_cost)
    assert best_cost == 4


def test_greedy_climb_constants(configuration_space_with_constants):
    best_config, best_results, best_cost = configuration_space_greedy_climb(
        configuration_space_with_constants,
        lambda config: dict(
            text=" ".join(
                [
                    config["determiner"],
                    config["animal"],
                    config["verb"],
                    config["activity"],
                ]
            )
        ),
        lambda results: 1.0
        - difflib.SequenceMatcher(None, results["text"], "A dog is barking").ratio(),
        random_sampler=lambda: Configuration(
            configuration_space_with_constants, vector=(0, 0, 0, 0)
        ),
    )
    print("Best configuration: ", best_config)
    print("Best results:", best_results)
    print("Best cost:", best_cost)
    assert best_results["text"] == "the dog is barking"
