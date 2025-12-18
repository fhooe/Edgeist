import struct

import numpy as np
import torch


class Datatype:
    # static Member
    dtype_map = {
        torch.float32: 0,
        torch.float16: 1,
        torch.float64: 2,
        torch.int32: 3,
        torch.int64: 4,
        # ... and so on
    }

    def __init__(self, type):
        self.type = self.dtype_map.get(type, None)

        if self.type is None:
            raise ValueError("Unknown Type")

    def get_number(self) -> int:
        return self.type

    @staticmethod
    def get_str(val) -> str:
        reverse_lookup = {v: k for k, v in Datatype.dtype_map.items()}
        return str(reverse_lookup.get(val)).split(".")[-1].strip() + "_t"

    @staticmethod
    def get_all_datatypes():
        return list(Datatype.dtype_map.values())
