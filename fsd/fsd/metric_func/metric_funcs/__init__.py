from .alpha_l1 import AlphaL1, RGBL1AlphaWeight
from .lpips import LPIPSAlphaWeight

metric_registry = {
    "AlphaL1": AlphaL1,
    "RGBL1AlphaWeight": RGBL1AlphaWeight,
    "LPIPSAlphaWeight": LPIPSAlphaWeight,
}
