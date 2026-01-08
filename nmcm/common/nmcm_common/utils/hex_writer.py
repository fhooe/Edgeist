from nmcm_common.utils.helper_functions import Sizeof
from nmcm_common.utils.hex_converter import HexConverter


class HexWriter:
    def __init__(self, filename: str):
        """
        Constructor of the class HEX_writter

        Checks fileending is valid .hex

        Parameters:
        filename (str): filename and path e.g. ./test.hex

        Raises:
        File is no .hex: If file ends not with .bin
        """

        # init json File
        self.filename = filename

        if self.filename.split(".")[-1] != "hex":
            raise ValueError(f"Hex-file '{self.filename}' is missing the '.hex' file extension")

        self.filename_trainable = self.filename.split(".")
        self.filename_trainable[len(self.filename_trainable) - 2] += "_trainable"
        self.filename_trainable = ".".join(self.filename_trainable)

    def writeHEX(self, model_struct):
        """
        Creates or clears the outputfile.
        writes all model information to the outputfile
        in hex format.

        Parameters:
        model_struct (array): list of header + all layers
        """
        offset = 0
        with open(self.filename_trainable, "wb") as outfile:
            for layer in model_struct:
                if layer.hex_trainable != b"":
                    # Get offset-Table type
                    offset_table_type = layer.config["Config-Info"]["Offset_Table"]
                    # update local offsets to global offsets
                    if layer.json_data["weightsAmountTrainable"] != 0:
                        pos = list(layer.dataorder.keys()).index("Weights_trainable")
                        offset_data = HexConverter(
                            offset + layer.json_data["weightsTrainableOffset"],
                            offset_table_type[1],
                        )
                        offset_pos = layer.Offset_Tabel_pos + pos * Sizeof(offset_table_type[1])
                        layer.hex_data = (
                            layer.hex_data[: layer.Offset_Table_pos + offset_pos]
                            + offset_data
                            + layer.hex_data[layer.Offset_Table_pos + offset_pos + len(offset_data) :]
                        )
                        layer.json_data["weightsTrainableOffset"] = offset + layer.json_data["weightsTrainableOffset"]

                    if layer.json_data["biasAmountTrainable"] != 0:
                        pos = list(layer.dataorder.keys()).index("biasTrainable")
                        offset_data = HexConverter(
                            offset + layer.json_data["biasTrainableOffset"],
                            offset_table_type[1],
                        )
                        offset_pos = layer.Offset_Tabel_pos + pos * Sizeof(offset_table_type[1])
                        layer.hex_data = (
                            layer.hex_data[: layer.Offset_Table_pos + offset_pos]
                            + offset_data
                            + layer.hex_data[layer.Offset_Table_pos + offset_pos + len(offset_data) :]
                        )
                        layer.json_data["biasTrainableOffset"] = offset + layer.json_data["biasTrainableOffset"]

                    # write trainable file
                    outfile.write(layer.hex_trainable)
                    offset += len(layer.hex_trainable)

        with open(self.filename, "wb") as outfile:
            for layer in model_struct:
                outfile.write(layer.hex_data)

        return model_struct
