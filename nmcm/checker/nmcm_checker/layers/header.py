from nmcm_common.utils import Sizeof

from nmcm_checker.layers.layer import Layer
from nmcm_checker.utils.binary_converter import BinaryConverter


class Header(Layer):
    def __init__(self):
        super().__init__()

    def reconstruct_data(self, data: str, config_data):
        # pure virtual function
        headersize_type = config_data["Header"]["Header_Size"]
        typeoffset = Sizeof(headersize_type)

        headersize = BinaryConverter(data[:typeoffset], headersize_type)
        header = data[:headersize]
        data = data[headersize:]

        for key, typename in config_data["Header"].items():
            datasize = Sizeof(typename)

            if key == "Version":
                version_len = len(config_data["configInfo"]["Version"])
                current_data = header[: datasize * version_len]
                self.json_data[key] = current_data.decode("ascii")
                header = header[datasize * version_len :]

            elif key == "offsetTable":
                self.json_data[key] = []
                for i in range(0, self.json_data["LayerNrs"]):
                    current_data = BinaryConverter(header[:datasize], typename)
                    self.json_data[key].append(current_data)
                    header = header[datasize:]
            else:
                current_data = BinaryConverter(header[:datasize], typename)
                self.json_data[key] = current_data
                header = header[datasize:]

        return data
