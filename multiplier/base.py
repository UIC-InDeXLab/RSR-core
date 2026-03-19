"""
Multiply M.v where v is a vector and M is available in the prep time.
"""

import torch


class Multiplier:
    def __init__(self, M: torch.Tensor):
        self.M = M

    def prep(self):
        raise NotImplementedError

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
