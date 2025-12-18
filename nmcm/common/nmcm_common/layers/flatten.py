from nmcm_common.layers.layer import Layer
from nmcm_common.utils.masks import Masks


class Flatten(Layer):
    def __init__(self, config):
        super().__init__(config)
        self.LayerId = 2

    def define_data(self, idx: int, modelinfo, masks: Masks = None):
        dimension = 1
        for i in range(1, len(modelinfo["input_shape"])):
            dimension *= modelinfo["input_shape"][i]

        # define all possible values of this layer without order
        self.data["LayerNr"] = idx
        self.data["ID"] = self.LayerId
        self.data["predecessorNr"] = 1
        self.data["predecessors"] = [idx - 1] if idx != 0 else [0]
        self.data["DimensionInput_x"] = dimension  # Input width
        self.data["DimensionOutput_x"] = dimension  # Output width

    def generate_data(self):
        if not (self.name in self.config):
            raise "Layer is not defined in config: " + self.name

        self._convert_config(self.config)
