import os
import random
from typing import Optional

import numpy as np
import torch


def seed_torch(seed: Optional[int] = 8402) -> None:
    """Set the seed for reproducibility across different libraries.

    Args:
        seed (Optional[int]): The seed value to use. Defaults to 1030.

    Returns:
        None
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
