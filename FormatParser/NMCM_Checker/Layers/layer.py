from abc import ABC, abstractmethod
from NMCN_Parser.Utils.helper_functions import Sizeof

from Utils.helperfunctions import BinaryConverter

import struct
import math


class Layer(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        self.json_data = {}

        # all keys that are pairs
        self.pairs = ["KernelSize", "padding", "stride", "dilation"]

    @abstractmethod
    def reconstruct_data(self, data: str, config_data):
        # pure virtual function
        pass

    def _read_offset_table(self, data, data_read):
        self.json_data["Offset_Table"] = []
        # read Offset Table
        datasize = Sizeof("uint32_t")
        for i in range(0, 7):
            current_data = data[:datasize]
            self.json_data["Offset_Table"].append(
                BinaryConverter(current_data, "uint32_t")
            )
            data = data[datasize:]
            data_read += datasize

        return data, data_read

    def _read_mask(self, data, name, length, data_read):
        bit_to_byte = 8
        data, data_read = self.align(data, data_read, Sizeof("uint32_t"))
        current_data = data[: math.ceil(length / bit_to_byte)]
        # Convert hex string to binary string (padded to 4 bits per hex digit)
        bit_array = "".join(f"{byte:08b}" for byte in current_data)

        bit_array = bit_array[:length]
        # Convert binary string to a list of bits (0 or 1)
        self.json_data[name + "_mask"] = [int(bit) for bit in bit_array]
        data = data[len(current_data) :]

        data_read += len(current_data)

        return data, data_read

    def _read_data(self, data, name, pos, data_read):
        datasize = Sizeof("float32_t")
        used = self.json_data[name]
        if self.json_data["Offset_Table"][pos] != 0:
            # trainable weights
            self.json_data[name + "_trainable"] = []
            data, data_read = self.align(data, data_read, datasize)
            if self.json_data["Offset_Table"][pos + 1] != 0:
                # split
                trainable_weights = (
                    self.json_data["Offset_Table"][pos + 1]
                    - self.json_data["Offset_Table"][pos]
                ) // datasize
                for idx in range(0, trainable_weights):
                    current_data = data[:datasize]
                    # get float value from hex string
                    self.json_data[name + "_trainable"].append(
                        BinaryConverter(current_data, "float32_t")
                    )
                    data = data[datasize:]
                    data_read += datasize
                used -= trainable_weights
            else:
                # only trainable
                for idx in range(0, self.json_data[name]):
                    current_data = data[:datasize]
                    # get float value from hex string
                    self.json_data[name + "_trainable"].append(
                        BinaryConverter(current_data, "float32_t")
                    )
                    data = data[datasize:]
                    data_read += datasize
                used = 0

        pos += 1
        if self.json_data["Offset_Table"][pos] != 0:
            # frozen weights
            self.json_data[name + "_frozen"] = []
            data, data_read = self.align(data, data_read, datasize)
            for idx in range(0, used):
                current_data = data[:datasize]
                # get float value from hex string
                self.json_data[name + "_frozen"].append(
                    BinaryConverter(current_data, "float32_t")
                )
                data = data[datasize:]
                data_read += datasize

        return data, pos, data_read

    def align(self, data, data_read, alignment):
        offset = alignment - data_read % alignment

        if offset != alignment:
            data = data[offset:]
            data_read += offset

        return data, data_read
