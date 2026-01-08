from collections import OrderedDict

from nmcm_common.utils.hex_converter import HexConverter


class Header:
    def __init__(self):
        self.data = {}
        self.Header_Size_info = (0, "")  # pos / type
        self.Offset_Table_info = (0, "")  # pos / type
        self.hex_data = b""
        self.hex_trainable = b""
        self.json_data = OrderedDict()
        self.name = "Header"

    def define_data(self, model_stuct):
        if len(model_stuct) == 0:
            raise RuntimeError("Model is empty")

        self.data["headerSize"] = 0  # placeholder
        self.data["magicNumber"] = 0x46434D4E
        self.data["version"] = "00.00.00"  # placeholder
        self.data["layerNrs"] = len(model_stuct)
        self.data["channelsIn"] = 1 if not ("channelsIn" in model_stuct[0].data) else model_stuct[0].data["channelsIn"]
        self.data["dimensionInputX"] = (
            1 if not ("dimensionInputX" in model_stuct[0].data) else model_stuct[0].data["dimensionInputX"]
        )
        self.data["dimensionInputY"] = (
            1 if not ("dimensionInputY" in model_stuct[0].data) else model_stuct[0].data["dimensionInputY"]
        )
        self.data["channelsOut"] = (
            1 if not ("channelsOut" in model_stuct[-1].data) else model_stuct[-1].data["channelsOut"]
        )
        self.data["dimensionOutputX"] = (
            1 if not ("dimensionOutputX" in model_stuct[-1].data) else model_stuct[-1].data["dimensionOutputX"]
        )
        self.data["dimensionOutputY"] = (
            1 if not ("dimensionOutputY" in model_stuct[-1].data) else model_stuct[-1].data["dimensionOutputY"]
        )
        self.data["layerOffsetTable"] = []

        offset = 0
        for layer in model_stuct:
            self.data["layerOffsetTable"].append(offset)
            offset += len(layer.hex_data)

    def generate_data(self, config):
        if not ("Config-Info" in config):
            raise RuntimeError("No 'Config-Info' section found config file")

        for key, value in config["Config-Info"].items():
            if key == "version":
                self.data["version"] = value[1]

        if not ("Header" in config):
            raise RuntimeError("No 'Header' section found in config file")

        for key, datatype in config["Header"].items():
            if not (key in self.data):
                raise RuntimeError(f"'Header' section missing key '{key}'")

            if key == "headerSize":
                self.Header_Size_info = (len(self.hex_data), datatype)
            if key == "layerOffsetTable":
                self.Offset_Table_info = (len(self.hex_data), datatype)

            # add data to json
            self.json_data[key] = self.data[key]

            # add data to hex
            self.hex_data += HexConverter(self.data[key], datatype[1])

        # Update Headersize
        header_len = len(self.hex_data)
        hex_header_len = HexConverter(header_len, self.Header_Size_info[1][1])
        self.json_data["headerSize"] = header_len
        self.hex_data = hex_header_len + self.hex_data[self.Header_Size_info[0] + len(hex_header_len) :]

        # get type offset
        offset_step = 0
        match self.Offset_Table_info[1][1]:
            case "uint32_t":
                # in byte
                offset_step = 4
            case _:
                raise ValueError(f"Unknown type: '{self.Offset_Table_info[1][1]}'")

        # Update Offset Table
        offset_pos = self.Offset_Table_info[0]
        idx = 0
        for elem in self.json_data["layerOffsetTable"]:
            elem += header_len
            self.json_data["layerOffsetTable"][idx] = elem
            new_offset = HexConverter(elem, self.Offset_Table_info[1][1])
            self.hex_data = self.hex_data[:offset_pos] + new_offset + self.hex_data[offset_pos + len(new_offset) :]
            offset_pos += offset_step
            idx += 1
