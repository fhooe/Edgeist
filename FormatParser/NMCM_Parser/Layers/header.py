from Utils.helper_functions import HexConverter
from collections import OrderedDict


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
            raise "Model is empty"

        self.data["Header_Size"] = 0  # placeholder
        self.data["Magic_Number"] = 0x46434D4E
        self.data["Version"] = "00.00.00"  # placeholder
        self.data["LayerNrs"] = len(model_stuct)
        self.data["ChannelsIn"] = (
            1
            if not ("ChannelsIn" in model_stuct[0].data)
            else model_stuct[0].data["ChannelsIn"]
        )
        self.data["DimensionInput_x"] = (
            1
            if not ("DimensionInput_x" in model_stuct[0].data)
            else model_stuct[0].data["DimensionInput_x"]
        )
        self.data["DimensionInput_y"] = (
            1
            if not ("DimensionInput_y" in model_stuct[0].data)
            else model_stuct[0].data["DimensionInput_y"]
        )
        self.data["ChannelsOut"] = (
            1
            if not ("ChannelsOut" in model_stuct[-1].data)
            else model_stuct[-1].data["ChannelsOut"]
        )
        self.data["DimensionOutput_x"] = (
            1
            if not ("DimensionOutput_x" in model_stuct[-1].data)
            else model_stuct[-1].data["DimensionOutput_x"]
        )
        self.data["DimensionOutput_y"] = (
            1
            if not ("DimensionOutput_y" in model_stuct[-1].data)
            else model_stuct[-1].data["DimensionOutput_y"]
        )
        self.data["Layer_Offset_Table"] = []

        offset = 0
        for layer in model_stuct:
            self.data["Layer_Offset_Table"].append(offset)
            offset += len(layer.hex_data)

    def generate_data(self, config):
        if not ("Config-Info" in config):
            raise "No Config-Info for Version in config file found"

        for key, value in config["Config-Info"].items():
            if key == "Version":
                self.data["Version"] = value[1]

        if not ("Header" in config):
            raise "No Header section in config file found"

        for key, datatype in config["Header"].items():
            if not (key in self.data):
                raise "Key: " + key + " is not defined in Header"

            if key == "Header_Size":
                self.Header_Size_info = (len(self.hex_data), datatype)
            if key == "Layer_Offset_Table":
                self.Offset_Table_info = (len(self.hex_data), datatype)

            # add data to json
            self.json_data[key] = self.data[key]

            # add data to hex
            self.hex_data += HexConverter(self.data[key], datatype[1])

        # Update Headersize
        header_len = len(self.hex_data)
        hex_header_len = HexConverter(header_len, self.Header_Size_info[1][1])
        self.json_data["Header_Size"] = header_len
        self.hex_data = (
            hex_header_len
            + self.hex_data[self.Header_Size_info[0] + len(hex_header_len) :]
        )

        # get type offset
        offset_step = 0
        match self.Offset_Table_info[1][1]:
            case "uint32_t":
                # in byte
                offset_step = 4
            case _:
                raise "Unknown Datatype: " + self.Offset_Table_info[1][1]

        # Update Offset Table
        offset_pos = self.Offset_Table_info[0]
        idx = 0
        for elem in self.json_data["Layer_Offset_Table"]:
            elem += header_len
            self.json_data["Layer_Offset_Table"][idx] = elem
            new_offset = HexConverter(elem, self.Offset_Table_info[1][1])
            self.hex_data = (
                self.hex_data[:offset_pos]
                + new_offset
                + self.hex_data[offset_pos + len(new_offset) :]
            )
            offset_pos += offset_step
            idx += 1
