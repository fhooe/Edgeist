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

import os
import sys
import numpy as np
import argparse
import torch

# inport Utils from ./Utils
from Utils.local_torchsummary import summary
from Utils.ConfigParser import Config_Parser
from Utils.datatyps import Masks
from Utils.JSONwriter import JSON_writer
from Utils.HEXwriter import HEX_writer
from Utils.Hwriter import H_writer
from Utils.helper_functions import update_header_offset_length

# import layers from ./Layers
from Layers.relu import ReLU
from Layers.conv2d import Conv2d
from Layers.linear import Linear
from Layers.maxpool2d import MaxPool2d
from Layers.softmax import Softmax
from Layers.header import Header
from Layers.flatten import Flatten
from Layers.batchnorm1d import BatchNorm1d
from Layers.batchnorm2d import BatchNorm2d
from Layers.adaptiveavgpool1d import AdaptiveAvgPool1d
from Layers.adaptiveavgpool2d import AdaptiveAvgPool2d
from Layers.dropout import Dropout

# import model class
# requirment for torch.load
from generatePytorchModel import SimpleNN


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
        json_writer = JSON_writer(filename + ".json")
        hex_writer = HEX_writer(filename + ".hex")
        h_writer = H_writer("./Modeltypes.h", "./Modelstructs.h", "./Modelenums.h")
        config = Config_Parser(configfile)
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

    #insert header at the front
    model_struct.insert(0,header)

    # update modular array-sizes
    configdata = update_header_offset_length(model_struct, configdata)

    # generate outputs
    model_struct = hex_writer.writeHEX(model_struct)
    json_writer.writeJSON(model_struct)
    h_writer.writeH(configdata, model_struct)

if __name__ == "__main__":
    # Possible arguments:
    #"args": [
    #            "--config", "./config.md",
    #            "--model", "./mnist_model.pth",
    #            "--inputsize", "(1,28,28)",
    #            "--name", "./model"
    #        ]

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Generate a json, hex and a h file from a pytorch model")

    # Add arguments
    parser.add_argument('--config', type=str, help='Path to configfile e.g. ./config.md')
    parser.add_argument('--model', type=str, help='Path to modelfile e.g. ./model.pth')
    parser.add_argument('--inputsize', type=str, help='Inputsize of the model e.g. (1,28,28)')
    parser.add_argument('--name', type=str, help='Name of the outputfiles')

    # Parse the arguments
    args = parser.parse_args()

    # convert inputsize str to tuple of int
    if not(str(args.inputsize).endswith(")")) or not(str(args.inputsize).startswith("(")):
        raise "Invalid Inputsize format"
    args.inputsize = str(args.inputsize).removeprefix("(").removesuffix(")")

    input = ()
    for value in str(args.inputsize).split(","):
        input = input + (int(value,10),)

    args.inputsize = input

    main(args)