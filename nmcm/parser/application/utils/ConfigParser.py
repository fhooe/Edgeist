import os
from collections import OrderedDict


class Config_Parser:
    def __init__(self, filename: str):
        if filename.split(".")[-1] != "md":
            raise ("Configfile is no .md")

        # remove output file if allready exists
        if not (os.path.exists(filename)):
            raise ("Configfile at: " + filename + " doesn't exists")

        self.configfile = filename

    def ReadConfigfile(self):
        configdata = {}
        currentSegment = ""

        with open(self.configfile, "r") as configfile:
            for line in configfile:
                # remove lineendings
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

                        if datatype == "Offset_Table_t":
                            for offset_name, offset_info in configdata["Offset_Table"].items():
                                configdata[currentSegment][offset_name] = offset_info
                        elif datatype == "Offset_Table":
                            configdata[currentSegment][name] = (amount, configdata["Config-Info"]["Offset_Table"][1])
                        else:
                            configdata[currentSegment][name] = (amount, datatype)

                    else:
                        # line is empty
                        pass

        # add ID at the beginning of every Layer
        for key, value in configdata["Config-Info"].items():
            if key == "ID":
                for name, data in configdata.items():
                    if name != "Config-Info" and name != "Offset_Table" and name != "Header":
                        data[key] = value
                        data.move_to_end(key, last=False)

        return configdata
