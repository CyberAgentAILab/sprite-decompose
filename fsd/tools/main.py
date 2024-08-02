import os.path as osp
import sys

import fsd
from fsd.engine.optimization import optimization
from fsd.engine.optimization_sample import optimization_sample
from fsd.utils.misc import seed_torch
from hydra import compose, initialize_config_dir

CONFIG_DIR = osp.join(fsd.__path__[0], "configs")


def parse_config_name(args: list[str]) -> tuple[str, list[str]]:
    config_name = [a for a in args if a.startswith("config_name=")]
    assert len(config_name) == 1, "'config_name' should be specified, like 'config_name=...'"
    args.pop(args.index(config_name[0]))
    config_name = config_name[0].split("=")[-1]
    return config_name, args


def main() -> None:
    args = sys.argv
    config_name, args = parse_config_name(args)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=config_name, overrides=args[1:])
    seed_torch(cfg.seed)

    if "optimization_sample" == cfg.command:
        optimization_sample(cfg)
    elif "optimization" == cfg.command:
        optimization(cfg)
    else:
        raise RuntimeError(f"Unknown command: {cfg.command}. Please choose from ['optimization_sample', 'optimization'].")


if __name__ == "__main__":
    main()
