#ifndef MODELSTRUCTS_H
#define MODELSTRUCTS_H

#include "Modeltypes.h"

// Header-Struct
#pragma pack(push, 1)
typedef struct{
	Header_Header_Size_t header_size;
	Header_Magic_Number_t magic_number;
	Header_Version_t version[8];
	Header_LayerNrs_t layernrs;
	Header_ChannelsIn_t channelsin;
	Header_DimensionInput_x_t dimensioninput_x;
	Header_DimensionInput_y_t dimensioninput_y;
	Header_ChannelsOut_t channelsout;
	Header_DimensionOutput_x_t dimensionoutput_x;
	Header_DimensionOutput_y_t dimensionoutput_y;
	Header_Layer_Offset_Table_t layer_offset_table[10];
} Neural_Network_Header_t;
#pragma pack(pop)

// Linear-Struct
#pragma pack(push, 1)
typedef struct{
	Linear_ID_t id;
	Linear_LayerNr_t layernr;
	Linear_predecessorNr_t predecessornr;
	Linear_DimensionInput_x_t dimensioninput_x;
	Linear_DimensionOutput_x_t dimensionoutput_x;
	Linear_Dataencoding_t dataencoding;
	Linear_Weights_amount_frozen_t weights_amount_frozen;
	Linear_Weights_amount_trainable_t weights_amount_trainable;
	Linear_Bias_amount_frozen_t bias_amount_frozen;
	Linear_Bias_amount_trainable_t bias_amount_trainable;
	Linear_Pruning_mask_offset_t pruning_mask_offset;
	Linear_Weights_mask_offset_t weights_mask_offset;
	Linear_Weights_trainable_offset_t weights_trainable_offset;
	Linear_Weights_frozen_offset_t weights_frozen_offset;
	Linear_Bias_mask_offset_t bias_mask_offset;
	Linear_Bias_trainable_offset_t bias_trainable_offset;
	Linear_Bias_frozen_offset_t bias_frozen_offset;
	Linear_predecessors_t* predecessors;
} Neural_Network_Linear_t;
#pragma pack(pop)

// Conv2d-Struct
#pragma pack(push, 1)
typedef struct{
	Conv2d_ID_t id;
	Conv2d_LayerNr_t layernr;
	Conv2d_predecessorNr_t predecessornr;
	Conv2d_DimensionInput_x_t dimensioninput_x;
	Conv2d_DimensionInput_y_t dimensioninput_y;
	Conv2d_DimensionOutput_x_t dimensionoutput_x;
	Conv2d_DimensionOutput_y_t dimensionoutput_y;
	Conv2d_ChannelsIn_t channelsin;
	Conv2d_ChannelsOut_t channelsout;
	Conv2d_KernelSize_t kernelsize[2];
	Conv2d_padding_t padding[2];
	Conv2d_stride_t stride[2];
	Conv2d_dilation_t dilation[2];
	Conv2d_groups_t groups;
	Conv2d_Dataencoding_t dataencoding;
	Conv2d_Weights_amount_frozen_t weights_amount_frozen;
	Conv2d_Weights_amount_trainable_t weights_amount_trainable;
	Conv2d_Bias_amount_frozen_t bias_amount_frozen;
	Conv2d_Bias_amount_trainable_t bias_amount_trainable;
	Conv2d_Pruning_mask_offset_t pruning_mask_offset;
	Conv2d_Weights_mask_offset_t weights_mask_offset;
	Conv2d_Weights_trainable_offset_t weights_trainable_offset;
	Conv2d_Weights_frozen_offset_t weights_frozen_offset;
	Conv2d_Bias_mask_offset_t bias_mask_offset;
	Conv2d_Bias_trainable_offset_t bias_trainable_offset;
	Conv2d_Bias_frozen_offset_t bias_frozen_offset;
	Conv2d_predecessors_t* predecessors;
} Neural_Network_Conv2d_t;
#pragma pack(pop)

// MaxPool2d-Struct
#pragma pack(push, 1)
typedef struct{
	MaxPool2d_ID_t id;
	MaxPool2d_LayerNr_t layernr;
	MaxPool2d_predecessorNr_t predecessornr;
	MaxPool2d_DimensionInput_x_t dimensioninput_x;
	MaxPool2d_DimensionInput_y_t dimensioninput_y;
	MaxPool2d_DimensionOutput_x_t dimensionoutput_x;
	MaxPool2d_DimensionOutput_y_t dimensionoutput_y;
	MaxPool2d_ChannelsIn_t channelsin;
	MaxPool2d_ChannelsOut_t channelsout;
	MaxPool2d_KernelSize_t kernelsize;
	MaxPool2d_padding_t padding;
	MaxPool2d_stride_t stride;
	MaxPool2d_dilation_t dilation;
	MaxPool2d_predecessors_t* predecessors;
} Neural_Network_MaxPool2d_t;
#pragma pack(pop)

// ReLU-Struct
#pragma pack(push, 1)
typedef struct{
	ReLU_ID_t id;
	ReLU_LayerNr_t layernr;
	ReLU_predecessorNr_t predecessornr;
	ReLU_DimensionInput_x_t dimensioninput_x;
	ReLU_DimensionOutput_x_t dimensionoutput_x;
	ReLU_predecessors_t* predecessors;
} Neural_Network_ReLU_t;
#pragma pack(pop)

// Softmax-Struct
#pragma pack(push, 1)
typedef struct{
	Softmax_ID_t id;
	Softmax_LayerNr_t layernr;
	Softmax_predecessorNr_t predecessornr;
	Softmax_DimensionInput_x_t dimensioninput_x;
	Softmax_DimensionOutput_x_t dimensionoutput_x;
	Softmax_predecessors_t* predecessors;
} Neural_Network_Softmax_t;
#pragma pack(pop)

// Flatten-Struct
#pragma pack(push, 1)
typedef struct{
	Flatten_ID_t id;
	Flatten_LayerNr_t layernr;
	Flatten_predecessorNr_t predecessornr;
	Flatten_DimensionInput_x_t dimensioninput_x;
	Flatten_DimensionOutput_x_t dimensionoutput_x;
	Flatten_predecessors_t* predecessors;
} Neural_Network_Flatten_t;
#pragma pack(pop)

// BatchNorm1d-Struct
#pragma pack(push, 1)
typedef struct{
	BatchNorm1d_ID_t id;
	BatchNorm1d_LayerNr_t layernr;
	BatchNorm1d_predecessorNr_t predecessornr;
	BatchNorm1d_DimensionInput_x_t dimensioninput_x;
	BatchNorm1d_DimensionOutput_x_t dimensionoutput_x;
	BatchNorm1d_Dataencoding_t dataencoding;
	BatchNorm1d_Weights_amount_frozen_t weights_amount_frozen;
	BatchNorm1d_Weights_amount_trainable_t weights_amount_trainable;
	BatchNorm1d_Bias_amount_frozen_t bias_amount_frozen;
	BatchNorm1d_Bias_amount_trainable_t bias_amount_trainable;
	BatchNorm1d_Pruning_mask_offset_t pruning_mask_offset;
	BatchNorm1d_Weights_mask_offset_t weights_mask_offset;
	BatchNorm1d_Weights_trainable_offset_t weights_trainable_offset;
	BatchNorm1d_Weights_frozen_offset_t weights_frozen_offset;
	BatchNorm1d_Bias_mask_offset_t bias_mask_offset;
	BatchNorm1d_Bias_trainable_offset_t bias_trainable_offset;
	BatchNorm1d_Bias_frozen_offset_t bias_frozen_offset;
	BatchNorm1d_predecessors_t* predecessors;
} Neural_Network_BatchNorm1d_t;
#pragma pack(pop)

// BatchNorm2d-Struct
#pragma pack(push, 1)
typedef struct{
	BatchNorm2d_ID_t id;
	BatchNorm2d_LayerNr_t layernr;
	BatchNorm2d_predecessorNr_t predecessornr;
	BatchNorm2d_DimensionInput_x_t dimensioninput_x;
	BatchNorm2d_DimensionOutput_x_t dimensionoutput_x;
	BatchNorm2d_ChannelsIn_t channelsin;
	BatchNorm2d_ChannelsOut_t channelsout;
	BatchNorm2d_Dataencoding_t dataencoding;
	BatchNorm2d_Weights_amount_frozen_t weights_amount_frozen;
	BatchNorm2d_Weights_amount_trainable_t weights_amount_trainable;
	BatchNorm2d_Bias_amount_frozen_t bias_amount_frozen;
	BatchNorm2d_Bias_amount_trainable_t bias_amount_trainable;
	BatchNorm2d_Pruning_mask_offset_t pruning_mask_offset;
	BatchNorm2d_Weights_mask_offset_t weights_mask_offset;
	BatchNorm2d_Weights_trainable_offset_t weights_trainable_offset;
	BatchNorm2d_Weights_frozen_offset_t weights_frozen_offset;
	BatchNorm2d_Bias_mask_offset_t bias_mask_offset;
	BatchNorm2d_Bias_trainable_offset_t bias_trainable_offset;
	BatchNorm2d_Bias_frozen_offset_t bias_frozen_offset;
	BatchNorm2d_predecessors_t* predecessors;
} Neural_Network_BatchNorm2d_t;
#pragma pack(pop)

// AdaptiveAvgPool1d-Struct
#pragma pack(push, 1)
typedef struct{
	AdaptiveAvgPool1d_ID_t id;
	AdaptiveAvgPool1d_LayerNr_t layernr;
	AdaptiveAvgPool1d_predecessorNr_t predecessornr;
	AdaptiveAvgPool1d_DimensionInput_x_t dimensioninput_x;
	AdaptiveAvgPool1d_DimensionOutput_x_t dimensionoutput_x;
	AdaptiveAvgPool1d_ChannelsIn_t channelsin;
	AdaptiveAvgPool1d_ChannelsOut_t channelsout;
	AdaptiveAvgPool1d_predecessors_t* predecessors;
} Neural_Network_AdaptiveAvgPool1d_t;
#pragma pack(pop)

// AdaptiveAvgPool2d-Struct
#pragma pack(push, 1)
typedef struct{
	AdaptiveAvgPool2d_ID_t id;
	AdaptiveAvgPool2d_LayerNr_t layernr;
	AdaptiveAvgPool2d_predecessorNr_t predecessornr;
	AdaptiveAvgPool2d_DimensionInput_x_t dimensioninput_x;
	AdaptiveAvgPool2d_DimensionInput_y_t dimensioninput_y;
	AdaptiveAvgPool2d_DimensionOutput_x_t dimensionoutput_x;
	AdaptiveAvgPool2d_DimensionOutput_y_t dimensionoutput_y;
	AdaptiveAvgPool2d_ChannelsIn_t channelsin;
	AdaptiveAvgPool2d_ChannelsOut_t channelsout;
	AdaptiveAvgPool2d_predecessors_t* predecessors;
} Neural_Network_AdaptiveAvgPool2d_t;
#pragma pack(pop)

// Dropout-Struct
#pragma pack(push, 1)
typedef struct{
	Dropout_ID_t id;
	Dropout_LayerNr_t layernr;
	Dropout_predecessorNr_t predecessornr;
	Dropout_DimensionInput_x_t dimensioninput_x;
	Dropout_DimensionOutput_x_t dimensionoutput_x;
	Dropout_DropoutRate_t dropoutrate;
	Dropout_predecessors_t* predecessors;
} Neural_Network_Dropout_t;
#pragma pack(pop)

#endif