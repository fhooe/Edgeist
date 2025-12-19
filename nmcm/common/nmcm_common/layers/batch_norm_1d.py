from nmcm_common.layers.layer import Layer
from nmcm_common.utils.datatype import Datatype
from nmcm_common.utils.helper_functions import Sizeof
from nmcm_common.utils.masks import Masks


class BatchNorm1d(Layer):
    def __init__(self, config):
        super().__init__(config)
        self.LayerId = 10
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
        self.data["ID"] = self.LayerId
        self.data["predecessorNr"] = 1
        self.data["predecessors"] = [idx - 1] if idx != 0 else [0]
        self.data["DimensionInput_x"] = modelinfo["input_shape"][1]  # Input width
        self.data["DimensionOutput_x"] = modelinfo["output_shape"][1]  # Output width
        self.data["Dataencoding"] = self.datatype.get_number()
        self.data["pruned"] = is_pruned

        # Append running stats
        running_mean = modelinfo.get("running_mean", [])
        running_var = modelinfo.get("running_var", [])

        self.data["Weights_amount_frozen"] = len(weights["frozen"]) if "frozen" in weights else 0
        self.data["Weights_amount_trainable"] = (len(weights["trainable"]) if "trainable" in weights else 0) + len(
            running_mean
        )
        self.data["Bias_amount_frozen"] = len(bias["frozen"]) if "frozen" in bias else 0
        self.data["Bias_amount_trainable"] = (len(bias["trainable"]) if "trainable" in bias else 0) + len(running_var)

        self.data["Pruning_mask_offset"] = 0
        self.data["Weights_mask_offset"] = 0
        self.data["Weights_trainable_offset"] = 0
        self.data["Weights_frozen_offset"] = 0
        self.data["Bias_mask_offset"] = 0
        self.data["Bias_trainable_offset"] = 0
        self.data["Bias_frozen_offset"] = 0

        if not (masks.Pruning_Mask is None):
            self.data["Pruning_mask"] = masks.Pruning_Mask.tolist()

        # Assemble weight lists
        if masks.Weight_Mask is not None:
            self.data["Weights_mask"] = masks.Weight_Mask.tolist()
            w_train = weights.get("trainable", [])
            w_frozen = weights.get("frozen", [])
        else:
            w_train = weights.get("trainable", [])
            w_frozen = weights.get("frozen", [])
        # Append running mean to trainable weights
        self.data["Weights_trainable"] = w_train + running_mean
        self.data["Weights_frozen"] = w_frozen

        # Assemble bias lists
        if masks.Bias_Mask is not None:
            self.data["Bias_mask"] = masks.Bias_Mask.tolist()
            b_train = bias.get("trainable", [])
            b_frozen = bias.get("frozen", [])
        else:
            b_train = bias.get("trainable", [])
            b_frozen = bias.get("frozen", [])
        # Append running var to trainable bias
        self.data["Bias_trainable"] = b_train + running_var
        self.data["Bias_frozen"] = b_frozen

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
