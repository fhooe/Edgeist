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

- .hex            Contains the model configuration and frozen weights and biases in a hex format
- _trainable.hex  Contains the model weights and biases that are trainable in a serialiced hex format
- .h files:       C header files defining model layers, types, and structures
- .json file:     (Optional) JSON representation of the model structure (currently disabled)

The script is intended for deployment of neural networks to embedded systems,
where PyTorch cannot be used directly.

Usage Example:
    python script.py
        --config ./config.md
        --model ./mnist_model.pth
        --inputsize "(1,28,28)"
        --name ./output/test_model

Inputs:
    --config     Path to the configuration file (e.g., ./config.md)
    --model      Path to the saved PyTorch model (e.g., .pth file)
    --inputsize  Input shape of the model as a tuple string (e.g., "(1,28,28)")
    --name       Base name for output files

Outputs:
    - <name>.hex
    - <name>_trainable.hex
    - Modeltypes.h
    - Modelstructs.h
    - Modelenums.h
    - (Optional) <name>.json

Notes:
    - The model must match the structure defined in `generatePytorchModel.py`
    - Only supported layers (defined in ./Layers) can be parsed
    - Configuration and conversion are driven by the `config.md` file

===============================================================================
"""

import argparse
import os
import sys

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


def main(args):
    # Input Parameter
    filename = args.name
    input_size = args.inputsize
    model_path = args.model
    configfile = args.config

    # load pytorch model
    model = torch.load(os.path.abspath(model_path), weights_only=False)

    # Construct all classes
    try:
        json_writer = JSONWriter(filename + ".json")
        hex_writer = HexWriter(filename + ".hex")
        h_writer = HeaderWriter("./Modeltypes.h", "./Modelstructs.h", "./Modelenums.h")
        config = ConfigParser(configfile)
        header = Header()
    except Exception as e:
        print(e)
        sys.exit(-1)

    # Parse config file
    configdata = config.ReadConfigfile()

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
            class_instance = dynamic_class(configdata)
            class_instance.define_data(idx, value, mask)
            class_instance.generate_data()
            model_struct.append(class_instance)

            print(f"[INFO] Conversion of Layer {idx+1}: {classname} completed")

        except Exception as e:
            print(f"[ERROR] Error converting Layer {idx+1} ({classname}): {e}")
            sys.exit(-1)

        idx += 1

    header.define_data(model_struct)
    header.generate_data(configdata)

    # insert header at the front
    model_struct.insert(0, header)

    # update modular array-sizes
    configdata = update_header_offset_length(model_struct, configdata)

    # generate outputs
    model_struct = hex_writer.writeHEX(model_struct)
    json_writer.writeJSON(model_struct)
    h_writer.writeH(configdata, model_struct)


if __name__ == "__main__":
    # Possible arguments:
    # "args": [
    #            "--config", "./config.md",
    #            "--model", "./mnist_model.pth",
    #            "--inputsize", "(1,28,28)",
    #            "--name", "./model"
    #        ]

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Generate json-, hex- and header-files from a pytorch model")

    # Add arguments
    parser.add_argument("--config", type=str, help="Path to config-file e.g. ./config.md")
    parser.add_argument("--model", type=str, help="Path to model-file e.g. ./model.pth")
    parser.add_argument("--inputsize", type=str, help="Input-size of the model e.g. (1,28,28)")
    parser.add_argument("--name", type=str, help="Name of the output-files")

    # Parse the arguments
    args = parser.parse_args()

    # convert inputsize str to tuple of int
    if not (str(args.inputsize).endswith(")")) or not (str(args.inputsize).startswith("(")):
        raise RuntimeError("Invalid input-size format")
    args.inputsize = str(args.inputsize).removeprefix("(").removesuffix(")")

    input = ()
    for value in str(args.inputsize).split(","):
        input = input + (int(value, 10),)

    args.inputsize = input

    main(args)
