import os
from collections import OrderedDict


class HeaderParser:
    def __init__(self, filename: str):
        if filename.split(".")[-1] != "h":
            raise ValueError(f"Header-file '{filename}' is missing the '.h' file extension")

        # remove output file if already exists
        if not (os.path.exists(filename)):
            raise FileNotFoundError(f"Header-file '{filename}' does not exist")

        self.configfile = filename

    def ReadHfile(self):
        config = {}
        current_config = ""
        with open(self.configfile, "r") as inputfile:
            lines = inputfile.readlines()
            for line in lines:
                line = line.strip()
                if line == "":
                    # empty line
                    continue

                if line.startswith("//"):
                    # comment to start new layer
                    current_config = line.split(" ")[-1].removesuffix("-Types")
                    config[current_config] = OrderedDict()
                    continue

                if line.startswith("#"):
                    # multiple inclusion protection or version
                    if line.startswith("#define version"):
                        config[current_config] = OrderedDict(
                            [
                                (
                                    "Version",
                                    line.split(" ")[-1].removeprefix('"').removesuffix('"'),
                                )
                            ]
                        )
                    continue

                config[current_config][line.split(" ")[-1].removeprefix(current_config + "_").removesuffix("_t;")] = (
                    line.split(" ")[1]
                )

        return config
