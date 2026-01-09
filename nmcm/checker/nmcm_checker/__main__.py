import os
import sys
from argparse import ArgumentParser

from nmcm_common.utils import JSONWriter, Sizeof

from nmcm_checker.layers import Conv2d, Header, Linear, MaxPool2d, ReLU
from nmcm_checker.utils import BinaryConverter, HeaderParser


def main(args):
    filename = args.name
    configfile = args.config

    json_writer = JSONWriter(r"./NMCN_Checker/test.json")
    model_struct = []

    # Read Config File
    conf = HeaderParser(configfile)
    config_data = conf.ReadHfile()

    # Read Hex File
    with open(filename, "rb") as hexfile:
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
    parser = ArgumentParser(description="Generates a json from a hex-file")

    # Add arguments
    parser.add_argument("--config", type=str, help="Path to configfile e.g. ./config.md")
    parser.add_argument("--name", type=str, help="Name of the input hex file")

    # Parse the arguments
    args = parser.parse_args()

    main(args)
