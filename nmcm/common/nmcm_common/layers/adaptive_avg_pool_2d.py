from nmcm_common.layers.layer import Layer
from nmcm_common.utils.masks import Masks


class AdaptiveAvgPool2d(Layer):
    #: The ID of the layer.
    LAYER_ID = 9

    def __init__(self, config):
        super().__init__(config)

    def define_data(self, idx: int, modelinfo, masks: Masks = None):
        self.data["LayerNr"] = idx
        self.data["ID"] = AdaptiveAvgPool2d.LAYER_ID
        self.data["predecessorNr"] = 1
        self.data["predecessors"] = [idx - 1] if idx != 0 else [0]
        self.data["DimensionInput_x"] = modelinfo["input_shape"][2]
        self.data["DimensionInput_y"] = modelinfo["input_shape"][3]
        self.data["DimensionOutput_x"] = modelinfo["output_shape"][2]
        self.data["DimensionOutput_y"] = modelinfo["output_shape"][3]
        self.data["ChannelsIn"] = modelinfo["input_shape"][1]
        self.data["ChannelsOut"] = modelinfo["output_shape"][1]

    def generate_data(self):
        if not (self.name in self.config):
            raise RuntimeError(f"Layer is not defined in config: '{self.name}'")

        self._convert_config(self.config)
