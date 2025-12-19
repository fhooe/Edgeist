import os
from abc import ABC, abstractmethod
from collections import OrderedDict

from nmcm_common.utils.datatype import Datatype
from nmcm_common.utils.helper_functions import Sizeof
from nmcm_common.utils.hex_converter import HexConverter
from nmcm_common.utils.masks import Masks


class Layer(ABC):
    def __init__(self, config):
        self.name = self.__class__.__name__
        self.hex_data = b""
        self.hex_trainable = b""
        self.json_data = OrderedDict()
        self.data = {}
        self.config = config

        # a empty type uses the type of the layer self.datatype.get_str()
        self.dataorder = OrderedDict(
            Pruning_mask="bit",
            Weights_mask="bit",
            Weights_trainable="",
            Weights_frozen="",
            Bias_mask="bit",
            Bias_trainable="",
            Bias_frozen="",
        )
        # dataalignement to 8 byte
        self.dataalignement = "uint32_t"

    @abstractmethod
    def define_data(self, idx: int, modelinfo, masks: Masks = None):
        # pure virtual function
        pass

    @abstractmethod
    def generate_data(self, config):
        # pure virtual function
        pass

    def _checkFilepath(self, filepath: str) -> bool:
        if filepath.split(".")[-1] != "json":
            return False

        if not (os.path.exists(filepath)):
            return False

        return True

    def _split_data(self, modelinfo, mask, name):
        data = {}
        flat_data = modelinfo[name]["data"].view(-1).numpy().tolist()

        if mask is None:
            if modelinfo[name]["trainable"]:
                data["trainable"] = flat_data
            else:
                data["frozen"] = flat_data
        else:
            data["trainable"] = []
            data["frozen"] = []
            for idx, value in enumerate(mask):
                if value == 0:
                    data["frozen"].append(flat_data[idx])
                else:
                    data["trainable"].append(flat_data[idx])

        return data

    def aligen(self, data, alignment: int):
        if len(data) % alignment != 0:
            data = data + b"\x00" * (alignment - len(data) % alignment)
        return data

    def _convert_config(self, config) -> int:
        offset_table_pos = 0

        for key, datatype_info in config[self.name].items():
            if not (key in self.data):
                raise RuntimeError(f"Key: '{key}' is not defined in Layer '{self.name}'")

            if key == next(iter(config["Offset_Table"])):
                offset_table_pos = len(self.hex_data)

            # add data to json
            self.json_data[key] = self.data[key]

            # add data to hex
            self.hex_data += HexConverter(self.data[key], datatype_info[1])

        return offset_table_pos

    def _convert_data(self, name: str, data_type: str, offset_tabel_pos: int, idx: int):
        if data_type == "" and hasattr(self, "datatype"):
            data_type = Datatype.get_str(self.datatype.get_number())

        # select bin file
        if "_trainable" in name.lower():
            data = self.hex_trainable
        else:
            data = self.hex_data

        if name in self.data:
            # write data
            data = self.aligen(data, Sizeof(self.dataalignement))
            offset = len(data)
            data += HexConverter(self.data[name], data_type)
            self.json_data[name] = self.data[name]

            if "_trainable" in name.lower():
                self.hex_trainable = data
            else:
                self.hex_data = data

            # write local offset
            offset_data = HexConverter(offset, self.config["Config-Info"]["Offset_Table"][1])
            self.hex_data = (
                self.hex_data[:offset_tabel_pos] + offset_data + self.hex_data[offset_tabel_pos + len(offset_data) :]
            )
            self.json_data[name + "_offset"] = offset
