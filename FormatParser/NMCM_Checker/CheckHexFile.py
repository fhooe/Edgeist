import sys
import os
import argparse

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NMCN_Parser.Utils.helper_functions import Sizeof
from NMCN_Parser.Utils.JSONwriter import JSON_writer
from Utils.helperfunctions import BinaryConverter

from Layers.conv2d import Conv2d
from Layers.header import Header
from Layers.linear import Linear
from Layers.maxpool2d import MaxPool2d
from Layers.relu import ReLU

from Utils.HParser import H_Parser

def main(args):
    filename = args.name
    configfile = args.config

    json_writer = JSON_writer(r"./NMCN_Checker/test.json")
    model_struct = []

    # Read Config File
    conf = H_Parser(configfile)
    config_data = conf.ReadHfile()

    # Read Hex File
    with open(filename,"rb") as hexfile:
        data = b''
        for line in hexfile:
            data += line

    # Read Header
    header = Header()
    data = header.reconstruct_data(data,config_data)
    model_struct.append(header)

    ID_size = Sizeof(config_data["Config-Info"]["ID"])

    # Read all Layer
    while data != b'':
        ID = BinaryConverter(data[:ID_size],config_data["Config-Info"]["ID"])

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
                raise "LayerID " + str(ID) + " is not implemented"
        
        data = layer.reconstruct_data(data,config_data)
        model_struct.append(layer)

    json_writer.writeJSON(model_struct)

if __name__ == "__main__":
    # possible arguments:
    #"args": [
    #            "--config", "./NMCN_Parser/Modeltypes.h",
    #            "--name", "./NMCN_Parser/test.hex"
    #        ]

    parser = argparse.ArgumentParser(description="Generate a json from a hex file")

    # Add arguments
    parser.add_argument('--config', type=str, help='Path to configfile e.g. ./config.md')
    parser.add_argument('--name', type=str, help='Name of the input hex file')

    # Parse the arguments
    args = parser.parse_args()

    main(args)