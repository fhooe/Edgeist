import os
from collections import OrderedDict


class ConfigParser:
    def __init__(self, filename: str):
        if filename.split(".")[-1] != "md":
            raise ValueError(f"Config-file '{filename}' is missing the '.md' file extension")

        # remove output file if already exists
        if not (os.path.exists(filename)):
            raise FileNotFoundError(f"Config-file '{filename}' does not exist")

        self.configfile = filename

    def ReadConfigfile(self):
        configdata = {}
        currentSegment = ""

        with open(self.configfile, "r") as configfile:
            for line in configfile:
                # remove line endings
                line = line.strip()

                # remove comments
                if line.startswith("##"):
                    continue

                if line.startswith("# "):
                    # new segment
                    currentSegment = line.split(" ")[-1]
                    configdata[currentSegment] = OrderedDict()
                else:
                    if line:
                        name = line.split(":")[0].split(" ")[-1]
                        datatype = line.split(":")[-1].split(" ")[-1]
                        amount_s = line.split(":")[-1].split("*")[0].strip()
                        amount = int(amount_s) if amount_s.isdigit() else 1

                        if datatype == "OffsetTable":
                            for offset_name, offset_info in configdata["offsetTable"].items():
                                configdata[currentSegment][offset_name] = offset_info
                        elif datatype == "offsetTable":
                            configdata[currentSegment][name] = (
                                amount,
                                configdata["configInfo"]["offsetTable"][1],
                            )
                        else:
                            configdata[currentSegment][name] = (amount, datatype)

                    else:
                        # line is empty
                        pass

        # add ID at the beginning of every Layer
        for key, value in configdata["configInfo"].items():
            if key == "ID":
                for name, data in configdata.items():
                    if name != "configInfo" and name != "offsetTable" and name != "Header":
                        data[key] = value
                        data.move_to_end(key, last=False)

        return configdata
