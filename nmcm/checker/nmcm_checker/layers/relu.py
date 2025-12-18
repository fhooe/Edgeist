from nmcm_common.utils import Sizeof

from nmcm_checker.layers.layer import Layer
from nmcm_checker.utils.binary_converter import BinaryConverter


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def reconstruct_data(self, data: str, config_data):
        # pure virtual function
        for key, typename in config_data[self.name].items():
            datasize = Sizeof(typename)

            # handle modular amount of predecessors
            if key == "predecessors":
                if not ("predecessorNr" in self.json_data):
                    raise "predecessors before predecessorNr in Layer: " + self.name

                current_data = []
                for i in range(0, self.json_data["predecessorNr"]):
                    current_data.append(BinaryConverter(data[:datasize], typename))
                    data = data[datasize:]
                self.json_data[key] = current_data
                continue

            current_data = data[:datasize]
            self.json_data[key] = BinaryConverter(data[:datasize], typename)
            data = data[datasize:]

        return data
