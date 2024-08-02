from typing import Any, Callable, Dict, Optional, Union

import torch.nn
from fsd.loss_func import loss_funcs


def build_loss_fn(name: str, params: Dict[str, Any]) -> Callable:
    if hasattr(loss_funcs, name):
        return getattr(loss_funcs, name)(**params)
    elif hasattr(torch.nn, name):
        return getattr(torch.nn, name)(**params)
    else:
        raise ValueError(f"{name} is not defined as a loss function.")


class LossFunctions:
    def __init__(self, loss_fns_spec: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        self.loss_fns = {}
        self.weights = {}
        self.use_target = {}
        self.labels = {}
        for key, specs in loss_fns_spec.items():
            for name, spec in specs.items():
                if spec["weight"] == 0:
                    continue
                self.loss_fns[f"{key}-{name}"] = build_loss_fn(name, spec["params"])
                self.weights[f"{key}-{name}"] = spec["weight"]
                self.use_target[f"{key}-{name}"] = spec["use_target"]
                self.labels[f"{key}-{name}"] = spec["label"]

    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        reduction: Optional[str] = None,
        labels: Optional[Any] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate losses for each output modalities.
        reduction: "sum" or None
        labels: list of labels to calculate
        """
        losses = {}
        for key, func in self.loss_fns.items():
            if labels is not None and self.labels[key] not in labels:
                continue
            modal = key.split("-")[0]
            if self.use_target[key]:
                loss = func(outputs[modal], targets[modal]) * self.weights[key]
            else:  # for regularization
                loss = func(outputs[modal]) * self.weights[key]
            losses[key] = loss
        if reduction == "sum":
            return sum(losses.values())
        elif reduction is None:
            return losses


def build_loss_funcs_from_cfg(cfg: Any) -> LossFunctions:
    return LossFunctions(cfg)
