import struct

import numpy as np
from nmcm_common.utils import Sizeof

from nmcm_checker.layers.layer import Layer
from nmcm_checker.utils.binary_converter import BinaryConverter


class Linear(Layer):
    def __init__(self):
        super().__init__()

    def reconstruct_data(self, data: str, config_data):
        # pure virtual function
        data_read = 0  # for alignment
        for key, typename in config_data[self.name].items():
            datasize = Sizeof(typename)

            # handle modular amount of predecessors
            if key == "predecessors":
                if not ("predecessorNr" in self.json_data):
                    raise RuntimeError(f"'predecessors' before 'predecessorNr' in layer '{self.name}'")

                current_data = []
                for i in range(0, self.json_data["predecessorNr"]):
                    current_data.append(BinaryConverter(data[:datasize], typename))
                    data = data[datasize:]
                    data_read += datasize
                self.json_data[key] = current_data
                continue

            # handle datapairs
            if key in self.pairs:
                current_data = [BinaryConverter(data[:datasize], typename)]
                data = data[datasize:]
                current_data.append(BinaryConverter(data[:datasize], typename))
                data = data[datasize:]
                self.json_data[key] = current_data
                data_read += datasize * 2
            else:
                # handle single entries
                current_data = data[:datasize]
                self.json_data[key] = BinaryConverter(data[:datasize], typename)
                data = data[datasize:]
                data_read += datasize

        data, data_read = self._read_offset_table(data, data_read)

        # read data
        pos = 0
        if self.json_data["Offset_Table"][pos] != 0:
            # pruning mask
            len_pruning_mask = self.json_data["Weights"] + self.json_data["Bias"]
            data, data_read = self._read_mask(data, "Pruning", len_pruning_mask, data_read)

        pos += 1
        if self.json_data["Offset_Table"][pos] != 0:
            # weight mask
            len_weight_mask = self.json_data["Weights"]
            data, data_read = self._read_mask(data, "Weights", len_weight_mask, data_read)

        pos += 1
        data, pos, data_read = self._read_data(data, "Weights", pos, data_read)

        pos += 1
        if self.json_data["Offset_Table"][pos] != 0:
            # bias mask
            len_bias_mask = self.json_data["Bias"]
            data, data_read = self._read_mask(data, "Bias", len_bias_mask, data_read)

        pos += 1
        data, pos, data_read = self._read_data(data, "Bias", pos, data_read)

        return data
