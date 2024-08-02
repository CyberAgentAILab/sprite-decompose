from typing import Any, Callable, Dict

from fsd.metric_func import metric_funcs


def build_metric_fn(name: str, params: Dict[str, Any]) -> Callable:
    if hasattr(metric_funcs, name):
        return getattr(metric_funcs, name)(**params)
    else:
        raise ValueError(f"{name} is not defined as a metric function.")


class MetricFunctions:
    def __init__(self, metric_fns_spec: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        self.metric_fns = {}
        self.goals = {}
        self.metric_fns_spec = metric_fns_spec
        for key, specs in metric_fns_spec.items():
            for name, spec in specs.items():
                if spec["activate"]:
                    self.metric_fns[f"{key}-{name}"] = build_metric_fn(name, spec["params"])
                    self.goals[f"{key}-{name}"] = spec["goal"]

    def __call__(self, outputs: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {}
        order = None
        for key, func in self.metric_fns.items():
            modal = key.split("-")[0]
            metrics[key], order = func(outputs[modal], targets[modal], order)
        return metrics


def build_metric_funcs_from_cfg(cfg: Any) -> MetricFunctions:
    return MetricFunctions(cfg)
