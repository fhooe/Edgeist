from nmcm_common.layers.layer import Layer
from nmcm_common.utils.masks import Masks


class AdaptiveAvgPool1d(Layer):
    #: The id of the layer.
    LAYER_ID = 8

    def __init__(self, config):
        super().__init__(config)

    def define_data(self, idx: int, modelinfo, masks: Masks = None):
        self.data["layerNr"] = idx
        self.data["id"] = AdaptiveAvgPool1d.LAYER_ID
        self.data["predecessorNr"] = 1
        self.data["predecessors"] = [idx - 1] if idx != 0 else [0]
        self.data["dimensionInputX"] = modelinfo["input_shape"][2]
        self.data["dimensionOutputX"] = modelinfo["output_shape"][2]
        self.data["channelsIn"] = modelinfo["input_shape"][1]
        self.data["channelsOut"] = modelinfo["output_shape"][1]

    def generate_data(self):
        if not (self.name in self.config):
            raise RuntimeError(f"Layer is not defined in config: '{self.name}'")

        self._convert_config(self.config)
