import argparse
import os
import sys

from nmcm_common.utils import JSONWriter, Sizeof

from nmcm_checker.layers import Conv2d, Header, Linear, MaxPool2d, ReLU
from nmcm_checker.utils import BinaryConverter, HeaderParser


def main(args):
    hex_file = args.hex_file
    model_types_header = args.model_types_header

    json_writer = JSONWriter("test.json")
    model_struct = []

    # Read Config File
    conf = HeaderParser(model_types_header)
    config_data = conf.ReadHfile()

    # Read Hex File
    with open(hex_file, "rb") as hexfile:
        data = b""
        for line in hexfile:
            data += line

    # Read Header
    header = Header()
    data = header.reconstruct_data(data, config_data)
    model_struct.append(header)

    ID_size = Sizeof(config_data["Config-Info"]["ID"])

    # Read all Layer
    while data != b"":
        ID = BinaryConverter(data[:ID_size], config_data["Config-Info"]["ID"])

        match ID:
            case 1:
                layer = Linear()
            case 3:
                layer = Conv2d()
            case 5:
                layer = MaxPool2d()
            case 6:
                layer = ReLU()
            case _:
                raise ValueError(f"Unknown layer {ID}")

        data = layer.reconstruct_data(data, config_data)
        model_struct.append(layer)

    json_writer.writeJSON(model_struct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a json from a hex file")

    # Add arguments
    parser.add_argument("--model_types_header", type=str, help="Path to the generated model_types.h file")
    parser.add_argument("--hex_file", type=str, help="Path to the generated model *.hex file")

    # Parse the arguments
    args = parser.parse_args()

    main(args)
