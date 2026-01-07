from nmcm_common.layers.layer import Layer
from nmcm_common.utils.datatype import Datatype
from nmcm_common.utils.helper_functions import Sizeof
from nmcm_common.utils.masks import Masks


class Linear(Layer):
    #: The ID of the layer.
    LAYER_ID = 1

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

        self.data["LayerNr"] = idx
        self.data["ID"] = Linear.LAYER_ID
        self.data["predecessorNr"] = 1
        self.data["predecessors"] = [idx - 1] if idx != 0 else [0]
        self.data["DimensionInput_x"] = modelinfo["input_shape"][1]  # Input width
        self.data["DimensionOutput_x"] = modelinfo["output_shape"][1]  # Output width
        self.data["Dataencoding"] = self.datatype.get_number()
        self.data["pruned"] = is_pruned
        self.data["Weights_amount_frozen"] = len(weights["frozen"]) if "frozen" in weights else 0
        self.data["Weights_amount_trainable"] = len(weights["trainable"]) if "trainable" in weights else 0
        self.data["Bias_amount_frozen"] = len(bias["frozen"]) if "frozen" in bias else 0
        self.data["Bias_amount_trainable"] = len(bias["trainable"]) if "trainable" in bias else 0

        self.data["Pruning_mask_offset"] = 0
        self.data["Weights_mask_offset"] = 0
        self.data["Weights_trainable_offset"] = 0
        self.data["Weights_frozen_offset"] = 0
        self.data["Bias_mask_offset"] = 0
        self.data["Bias_trainable_offset"] = 0
        self.data["Bias_frozen_offset"] = 0

        if not (masks.Pruning_Mask is None):
            self.data["Pruning_mask"] = masks.Pruning_Mask.tolist()

        if not (masks.Weight_Mask is None):
            self.data["Weights_mask"] = masks.Weight_Mask.tolist()
            self.data["Weights_trainable"] = weights["trainable"]
            self.data["Weights_frozen"] = weights["frozen"]
        else:
            if "trainable" in weights:
                self.data["Weights_trainable"] = weights["trainable"]
            if "frozen" in weights:
                self.data["Weights_frozen"] = weights["frozen"]

        if not (masks.Bias_Mask is None):
            self.data["Bias_mask"] = masks.Bias_Mask.tolist()
            self.data["Bias_trainable"] = bias["trainable"]
            self.data["Bias_frozen"] = bias["frozen"]
        else:
            if "trainable" in bias:
                self.data["Bias_trainable"] = bias["trainable"]
            if "frozen" in bias:
                self.data["Bias_frozen"] = bias["frozen"]

    def generate_data(
        self,
    ):
        if not (self.name in self.config):
            raise RuntimeError(f"Layer is not defined in config: '{self.name}'")

        offset_table_pos = self._convert_config(self.config)

        ########################################################################
        # add Data segment
        ########################################################################

        # add weight and bias
        # used to update relative pos to absolute file pos later
        self.Offset_Table_pos = offset_table_pos

        idx = 0
        for name, type in self.dataorder.items():
            self._convert_data(name, type, offset_table_pos, idx)
            # Size of one entry in the offset table
            offset_table_pos += Sizeof(self.config["Config-Info"]["Offset_Table"][1])
            idx += 1
