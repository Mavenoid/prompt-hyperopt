from typing import Any, Dict, Union, List

import ConfigSpace


def make_configuration_space_from_option_dict(
    options: Dict[str, Union[Any, List[Any]]],
) -> ConfigSpace.ConfigurationSpace:
    # @TODO streamline
    cs = ConfigSpace.ConfigurationSpace(seed=0)
    # index path; conditions; subvariations
    stack = [([], [], options)]
    while stack:
        path, conds, ent = stack.pop()
        if isinstance(ent, dict):
            for k, v in ent.items():
                stack.append((path + [k], conds, v))
        elif isinstance(ent, list) or isinstance(ent, tuple):
            if len(ent) == 0:
                continue
            elif len(ent) == 1:
                if isinstance(ent[0], str):
                    hp = ConfigSpace.hyperparameters.Constant(
                        name="__".join(path),
                        value=ent[0],
                        meta=dict(path=path),
                    )
                else:
                    hp = ConfigSpace.hyperparameters.Constant(
                        name="__".join(path),
                        value=0,
                        meta=dict(path=path, values=ent),
                    )
            else:
                choices = [x if isinstance(x, str) else i for i, x in enumerate(ent)]
                hp = ConfigSpace.hyperparameters.CategoricalHyperparameter(
                    name="__".join(path),
                    choices=choices,
                    meta=dict(path=path, values=ent),
                )
            cs.add_hyperparameter(hp)
            for cond in conds:
                cd = ConfigSpace.conditions.EqualsCondition(hp, cond[0], cond[1])
                cs.add_condition(cd)
            for idx, val in enumerate(ent):
                if (
                    isinstance(val, dict)
                    or isinstance(val, list)
                    or isinstance(val, tuple)
                ):
                    stack.append((path + [str(idx)], conds + [(hp, idx)], val))
        else:
            if isinstance(ent, str):
                hp = ConfigSpace.hyperparameters.Constant(
                    name="____".join(path),
                    value=ent,
                    meta=dict(path=path),
                )
            else:
                hp = ConfigSpace.hyperparameters.Constant(
                    name="____".join(path),
                    value=0,
                    meta=dict(path=path, values=[ent]),
                )
            cs.add_hyperparameter(hp)
            for cond in conds:
                cd = ConfigSpace.conditions.EqualsCondition(hp, cond[0], cond[1])
                cs.add_condition(cd)

    return cs
