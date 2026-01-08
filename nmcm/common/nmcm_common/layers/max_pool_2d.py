from nmcm_common.layers.layer import Layer
from nmcm_common.utils.masks import Masks


class MaxPool2d(Layer):
    #: The ID of the layer.
    LAYER_ID = 5

    def __init__(self, config):
        super().__init__(config)

    def define_data(self, idx: int, modelinfo, masks: Masks = None):
        self.data["layerNr"] = idx
        self.data["id"] = MaxPool2d.LAYER_ID
        self.data["predecessorNr"] = 1
        self.data["predecessors"] = [idx - 1] if idx != 0 else [0]
        self.data["dimensionInputX"] = modelinfo["input_shape"][2]
        self.data["dimensionInputY"] = modelinfo["input_shape"][3]
        self.data["dimensionOutputX"] = modelinfo["output_shape"][2]
        self.data["dimensionOutputY"] = modelinfo["output_shape"][3]
        self.data["channelsIn"] = modelinfo["input_shape"][1]
        self.data["channelsOut"] = modelinfo["output_shape"][1]
        self.data["kernelSize"] = modelinfo["kernel_size"]
        self.data["padding"] = modelinfo["padding"]
        self.data["stride"] = modelinfo["stride"]
        self.data["dilation"] = modelinfo["dilation"]

    def generate_data(self):
        if not (self.name in self.config):
            raise RuntimeError(f"Layer is not defined in config: '{self.name}'")

        self._convert_config(self.config)
