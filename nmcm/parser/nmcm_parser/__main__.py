"""
===============================================================================
PyTorch Model Converter Script
-------------------------------------------------------------------------------
Author: Matthias Koehler, David Muttenthaler
Date: 24.06.2025
Python Version: 3.x
Dependencies: torch, numpy, argparse, custom Utils and Layers modules
-------------------------------------------------------------------------------

Description:
This script converts a PyTorch model into a format that is compatible with the Neural MicroController Framework (NMCF).
It extracts model structure and weights, processes each layer
using custom classes, and generates the following outputs:

- model.hex                         Contains the model configuration and frozen weights and biases in a hex format
- model_trainable.hex               Contains the model weights and biases that are trainable in a serialized hex format
- model_[types | enums | structs].h C++ header files defining model layers, types, and structures
- model.json file:                  JSON representation of the model structure

The script is intended for deployment of neural networks to embedded systems,
where PyTorch cannot be used directly.

Notes:
    - The model must match the structure defined in `simple_nn.py`
    - Only supported layers can be parsed
    - Configuration and conversion are driven by the `config.md` file

===============================================================================
"""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from nmcm_common.layers import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Softmax,
)
from nmcm_common.nn import SimpleNN
from nmcm_common.utils import (
    ConfigParser,
    Header,
    HeaderWriter,
    HexWriter,
    JSONWriter,
    Masks,
    summary,
    update_header_offset_length,
)


def main(args) -> None:
    # Input Parameter
    input_size = args.input_size
    model = args.model
    config = args.config

    # load pytorch model
    model = torch.load(os.path.abspath(model), weights_only=False)

    # Construct all classes
    try:
        json_writer = JSONWriter(os.path.join(args.hex_out_dir, "model.json"))
        hex_writer = HexWriter(os.path.join(args.hex_out_dir, "model.hex"))
        header_writer = HeaderWriter(
            os.path.join(args.header_out_dir, "model_types.h"),
            os.path.join(args.header_out_dir, "model_structs.h"),
            os.path.join(args.header_out_dir, "model_enums.h"),
        )
        config_parser = ConfigParser(config)
        header = Header()
    except Exception as e:
        print(e)
        sys.exit(-1)

    # Parse config file
    config_data = config_parser.ReadConfigfile()

    # Variable from optimizer
    mask = Masks()

    # generate Model summary
    layers = summary(model, input_size=input_size)
    idx = 0

    # Construct Layers
    model_struct = []
    for key, value in layers.items():
        classname = str(key).split("-")[0]

        print(f"[INFO] Conversion of Layer {idx+1}: {classname} started...")

        # look for class by name
        try:
            dynamic_class = globals()[classname]
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(-1)

        mask.Weight_Mask = None

        try:
            # Create correct layer class
            class_instance = dynamic_class(config_data)
            class_instance.define_data(idx, value, mask)
            class_instance.generate_data()
            model_struct.append(class_instance)

            print(f"[INFO] Conversion of Layer {idx+1}: {classname} completed")

        except Exception as e:
            print(f"[ERROR] Error converting Layer {idx+1} ({classname}): {e}")
            sys.exit(-1)

        idx += 1

    header.define_data(model_struct)
    header.generate_data(config_data)

    # insert header at the front
    model_struct.insert(0, header)

    # update modular array-sizes
    config_data = update_header_offset_length(model_struct, config_data)

    # generate outputs
    model_struct = hex_writer.writeHEX(model_struct)
    json_writer.writeJSON(model_struct)
    header_writer.write(config_data)


if __name__ == "__main__":
    # Create the argument parser
    parser = ArgumentParser(description="Generates json-, hex- and header-files from a pytorch models")

    # Add arguments
    parser.add_argument("--config", type=str, help="path to the config-file", default="config.md")
    parser.add_argument("--model", type=str, help="path to the model", required=True)
    parser.add_argument("--input_size", type=str, help="input-size of the model", default="(1,28,28)")
    parser.add_argument(
        "--header_out_dir", type=str, help="destination directory of generated header-files", default="output"
    )
    parser.add_argument(
        "--hex_out_dir", type=str, help="destination directory of generated hex- (+json-) files", default="output"
    )

    # Parse the arguments
    args = parser.parse_args()

    # convert inputsize str to tuple of int
    if not (str(args.input_size).endswith(")")) or not (str(args.input_size).startswith("(")):
        raise RuntimeError("Invalid input-size format")
    args.input_size = str(args.input_size).removeprefix("(").removesuffix(")")

    input = ()
    for value in str(args.input_size).split(","):
        input = input + (int(value, 10),)

    args.input_size = input

    main(args)
