from nmcm_common.layers.layer import Layer
from nmcm_common.utils.datatype import Datatype
from nmcm_common.utils.helper_functions import Sizeof
from nmcm_common.utils.masks import Masks


class Conv2d(Layer):
    #: The ID of the layer.
    LAYER_ID = 3

    def __init__(self, config):
        super().__init__(config)
        self.datatype = object
        self.Offset_Tabel_pos = 0

    def define_data(self, idx: int, modelinfo, masks: Masks = None):
        weights = self._split_data(modelinfo, masks.Weight_Mask, "weights")
        bias = self._split_data(modelinfo, masks.Bias_Mask, "bias")

        self.datatype = Datatype(modelinfo["weights"]["dtype"])

        is_pruned = False
        if not (masks.Pruning_Mask is None):
            is_pruned = True

        self.data["layerNr"] = idx
        self.data["id"] = Conv2d.LAYER_ID
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
        self.data["groups"] = modelinfo["groups"]
        self.data["dataEncoding"] = self.datatype.get_number()
        self.data["pruned"] = is_pruned
        self.data["weightsAmountFrozen"] = len(weights["frozen"]) if "frozen" in weights else 0
        self.data["weightsAmountTrainable"] = len(weights["trainable"]) if "trainable" in weights else 0
        self.data["biasAmountFrozen"] = len(bias["frozen"]) if "frozen" in bias else 0
        self.data["biasAmountTrainable"] = len(bias["trainable"]) if "trainable" in bias else 0

        self.data["pruningMaskOffset"] = 0
        self.data["weightsMaskOffset"] = 0
        self.data["weightsTrainableOffset"] = 0
        self.data["weightsFrozenOffset"] = 0
        self.data["biasMaskOffset"] = 0
        self.data["biasTrainableOffset"] = 0
        self.data["biasFrozenOffset"] = 0

        if not (masks.Pruning_Mask is None):
            self.data["pruningMask"] = masks.Pruning_Mask.tolist()

        if not (masks.Weight_Mask is None):
            self.data["weightsMask"] = masks.Weight_Mask.tolist()
            self.data["weightsTrainable"] = weights["trainable"]
            self.data["weightsFrozen"] = weights["frozen"]
        else:
            if "trainable" in weights:
                self.data["weightsTrainable"] = weights["trainable"]
            if "frozen" in weights:
                self.data["weightsFrozen"] = weights["frozen"]

        if not (masks.Bias_Mask is None):
            self.data["biasMask"] = masks.Bias_Mask.tolist()
            self.data["biasTrainable"] = bias["trainable"]
            self.data["biasFrozen"] = bias["frozen"]
        else:
            if "trainable" in bias:
                self.data["biasTrainable"] = bias["trainable"]
            if "frozen" in bias:
                self.data["biasFrozen"] = bias["frozen"]

    def generate_data(self):
        if not (self.name in self.config):
            raise RuntimeError(f"Layer is not defined in config: '{self.name}'")

        offset_table_pos = self._convert_config(self.config)

        ########################################################################
        # add Data segment
        ########################################################################

        # add weight and bias
        # used to update relative pos to absolute file pos later
        self.offsetTable_pos = offset_table_pos

        idx = 0
        for name, type in self.dataorder.items():
            self._convert_data(name, type, offset_table_pos, idx)
            # Size of one entry in the offset table
            offset_table_pos += Sizeof(self.config["configInfo"]["offsetTable"][1])
            idx += 1
