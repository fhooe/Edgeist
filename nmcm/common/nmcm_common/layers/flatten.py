from nmcm_common.layers.layer import Layer
from nmcm_common.utils.masks import Masks


class Flatten(Layer):
    #: The id of the layer.
    LAYER_ID = 2

    def __init__(self, config):
        super().__init__(config)

    def define_data(self, idx: int, modelinfo, masks: Masks = None):
        dimension = 1
        for i in range(1, len(modelinfo["input_shape"])):
            dimension *= modelinfo["input_shape"][i]

        # define all possible values of this layer without order
        self.data["layerNr"] = idx
        self.data["id"] = Flatten.LAYER_ID
        self.data["predecessorNr"] = 1
        self.data["predecessors"] = [idx - 1] if idx != 0 else [0]
        self.data["dimensionInputX"] = dimension  # Input width
        self.data["dimensionOutputX"] = dimension  # Output width

    def generate_data(self):
        if not (self.name in self.config):
            raise RuntimeError(f"Layer is not defined in config: '{self.name}'")

        self._convert_config(self.config)
