#ifndef MODELTYPES_H
#define MODELTYPES_H

#include<stdint.h>

// Config-Info-Types
#define version_str "01.00.00"
typedef uint8_t ID_t;
typedef uint32_t Offset_Table_entry;

// Header-Types
	typedef uint16_t Header_Header_Size_t;
	typedef uint32_t Header_Magic_Number_t;
	typedef uint8_t Header_Version_t;
	typedef uint16_t Header_LayerNrs_t;
	typedef uint16_t Header_ChannelsIn_t;
	typedef uint32_t Header_DimensionInput_x_t;
	typedef uint32_t Header_DimensionInput_y_t;
	typedef uint16_t Header_ChannelsOut_t;
	typedef uint32_t Header_DimensionOutput_x_t;
	typedef uint32_t Header_DimensionOutput_y_t;
	typedef uint32_t Header_Layer_Offset_Table_t;

// Linear-Types
	typedef uint8_t Linear_ID_t;
	typedef uint8_t Linear_LayerNr_t;
	typedef uint8_t Linear_predecessorNr_t;
	typedef uint32_t Linear_DimensionInput_x_t;
	typedef uint32_t Linear_DimensionOutput_x_t;
	typedef uint8_t Linear_Dataencoding_t;
	typedef uint32_t Linear_Weights_amount_frozen_t;
	typedef uint32_t Linear_Weights_amount_trainable_t;
	typedef uint16_t Linear_Bias_amount_frozen_t;
	typedef uint32_t Linear_Bias_amount_trainable_t;
	typedef uint32_t Linear_Pruning_mask_offset_t;
	typedef uint32_t Linear_Weights_mask_offset_t;
	typedef uint32_t Linear_Weights_trainable_offset_t;
	typedef uint32_t Linear_Weights_frozen_offset_t;
	typedef uint32_t Linear_Bias_mask_offset_t;
	typedef uint32_t Linear_Bias_trainable_offset_t;
	typedef uint32_t Linear_Bias_frozen_offset_t;
	typedef uint16_t Linear_predecessors_t;

// Conv2d-Types
	typedef uint8_t Conv2d_ID_t;
	typedef uint8_t Conv2d_LayerNr_t;
	typedef uint8_t Conv2d_predecessorNr_t;
	typedef uint32_t Conv2d_DimensionInput_x_t;
	typedef uint32_t Conv2d_DimensionInput_y_t;
	typedef uint32_t Conv2d_DimensionOutput_x_t;
	typedef uint32_t Conv2d_DimensionOutput_y_t;
	typedef uint16_t Conv2d_ChannelsIn_t;
	typedef uint16_t Conv2d_ChannelsOut_t;
	typedef uint8_t Conv2d_KernelSize_t;
	typedef uint8_t Conv2d_padding_t;
	typedef uint8_t Conv2d_stride_t;
	typedef uint8_t Conv2d_dilation_t;
	typedef uint8_t Conv2d_groups_t;
	typedef uint8_t Conv2d_Dataencoding_t;
	typedef uint32_t Conv2d_Weights_amount_frozen_t;
	typedef uint32_t Conv2d_Weights_amount_trainable_t;
	typedef uint16_t Conv2d_Bias_amount_frozen_t;
	typedef uint32_t Conv2d_Bias_amount_trainable_t;
	typedef uint32_t Conv2d_Pruning_mask_offset_t;
	typedef uint32_t Conv2d_Weights_mask_offset_t;
	typedef uint32_t Conv2d_Weights_trainable_offset_t;
	typedef uint32_t Conv2d_Weights_frozen_offset_t;
	typedef uint32_t Conv2d_Bias_mask_offset_t;
	typedef uint32_t Conv2d_Bias_trainable_offset_t;
	typedef uint32_t Conv2d_Bias_frozen_offset_t;
	typedef uint16_t Conv2d_predecessors_t;

// MaxPool2d-Types
	typedef uint8_t MaxPool2d_ID_t;
	typedef uint8_t MaxPool2d_LayerNr_t;
	typedef uint8_t MaxPool2d_predecessorNr_t;
	typedef uint32_t MaxPool2d_DimensionInput_x_t;
	typedef uint32_t MaxPool2d_DimensionInput_y_t;
	typedef uint32_t MaxPool2d_DimensionOutput_x_t;
	typedef uint32_t MaxPool2d_DimensionOutput_y_t;
	typedef uint16_t MaxPool2d_ChannelsIn_t;
	typedef uint16_t MaxPool2d_ChannelsOut_t;
	typedef uint8_t MaxPool2d_KernelSize_t;
	typedef uint8_t MaxPool2d_padding_t;
	typedef uint8_t MaxPool2d_stride_t;
	typedef uint8_t MaxPool2d_dilation_t;
	typedef uint16_t MaxPool2d_predecessors_t;

// ReLU-Types
	typedef uint8_t ReLU_ID_t;
	typedef uint8_t ReLU_LayerNr_t;
	typedef uint8_t ReLU_predecessorNr_t;
	typedef uint32_t ReLU_DimensionInput_x_t;
	typedef uint32_t ReLU_DimensionOutput_x_t;
	typedef uint16_t ReLU_predecessors_t;

// Softmax-Types
	typedef uint8_t Softmax_ID_t;
	typedef uint8_t Softmax_LayerNr_t;
	typedef uint8_t Softmax_predecessorNr_t;
	typedef uint32_t Softmax_DimensionInput_x_t;
	typedef uint32_t Softmax_DimensionOutput_x_t;
	typedef uint16_t Softmax_predecessors_t;

// Flatten-Types
	typedef uint8_t Flatten_ID_t;
	typedef uint8_t Flatten_LayerNr_t;
	typedef uint8_t Flatten_predecessorNr_t;
	typedef uint32_t Flatten_DimensionInput_x_t;
	typedef uint32_t Flatten_DimensionOutput_x_t;
	typedef uint16_t Flatten_predecessors_t;

// BatchNorm1d-Types
	typedef uint8_t BatchNorm1d_ID_t;
	typedef uint8_t BatchNorm1d_LayerNr_t;
	typedef uint8_t BatchNorm1d_predecessorNr_t;
	typedef uint32_t BatchNorm1d_DimensionInput_x_t;
	typedef uint32_t BatchNorm1d_DimensionOutput_x_t;
	typedef uint8_t BatchNorm1d_Dataencoding_t;
	typedef uint32_t BatchNorm1d_Weights_amount_frozen_t;
	typedef uint32_t BatchNorm1d_Weights_amount_trainable_t;
	typedef uint16_t BatchNorm1d_Bias_amount_frozen_t;
	typedef uint32_t BatchNorm1d_Bias_amount_trainable_t;
	typedef uint32_t BatchNorm1d_Pruning_mask_offset_t;
	typedef uint32_t BatchNorm1d_Weights_mask_offset_t;
	typedef uint32_t BatchNorm1d_Weights_trainable_offset_t;
	typedef uint32_t BatchNorm1d_Weights_frozen_offset_t;
	typedef uint32_t BatchNorm1d_Bias_mask_offset_t;
	typedef uint32_t BatchNorm1d_Bias_trainable_offset_t;
	typedef uint32_t BatchNorm1d_Bias_frozen_offset_t;
	typedef uint16_t BatchNorm1d_predecessors_t;

// BatchNorm2d-Types
	typedef uint8_t BatchNorm2d_ID_t;
	typedef uint8_t BatchNorm2d_LayerNr_t;
	typedef uint8_t BatchNorm2d_predecessorNr_t;
	typedef uint32_t BatchNorm2d_DimensionInput_x_t;
	typedef uint32_t BatchNorm2d_DimensionOutput_x_t;
	typedef uint16_t BatchNorm2d_ChannelsIn_t;
	typedef uint16_t BatchNorm2d_ChannelsOut_t;
	typedef uint8_t BatchNorm2d_Dataencoding_t;
	typedef uint32_t BatchNorm2d_Weights_amount_frozen_t;
	typedef uint32_t BatchNorm2d_Weights_amount_trainable_t;
	typedef uint16_t BatchNorm2d_Bias_amount_frozen_t;
	typedef uint32_t BatchNorm2d_Bias_amount_trainable_t;
	typedef uint32_t BatchNorm2d_Pruning_mask_offset_t;
	typedef uint32_t BatchNorm2d_Weights_mask_offset_t;
	typedef uint32_t BatchNorm2d_Weights_trainable_offset_t;
	typedef uint32_t BatchNorm2d_Weights_frozen_offset_t;
	typedef uint32_t BatchNorm2d_Bias_mask_offset_t;
	typedef uint32_t BatchNorm2d_Bias_trainable_offset_t;
	typedef uint32_t BatchNorm2d_Bias_frozen_offset_t;
	typedef uint16_t BatchNorm2d_predecessors_t;

// AdaptiveAvgPool1d-Types
	typedef uint8_t AdaptiveAvgPool1d_ID_t;
	typedef uint8_t AdaptiveAvgPool1d_LayerNr_t;
	typedef uint8_t AdaptiveAvgPool1d_predecessorNr_t;
	typedef uint32_t AdaptiveAvgPool1d_DimensionInput_x_t;
	typedef uint32_t AdaptiveAvgPool1d_DimensionOutput_x_t;
	typedef uint16_t AdaptiveAvgPool1d_ChannelsIn_t;
	typedef uint16_t AdaptiveAvgPool1d_ChannelsOut_t;
	typedef uint16_t AdaptiveAvgPool1d_predecessors_t;

// AdaptiveAvgPool2d-Types
	typedef uint8_t AdaptiveAvgPool2d_ID_t;
	typedef uint8_t AdaptiveAvgPool2d_LayerNr_t;
	typedef uint8_t AdaptiveAvgPool2d_predecessorNr_t;
	typedef uint32_t AdaptiveAvgPool2d_DimensionInput_x_t;
	typedef uint32_t AdaptiveAvgPool2d_DimensionInput_y_t;
	typedef uint32_t AdaptiveAvgPool2d_DimensionOutput_x_t;
	typedef uint32_t AdaptiveAvgPool2d_DimensionOutput_y_t;
	typedef uint16_t AdaptiveAvgPool2d_ChannelsIn_t;
	typedef uint16_t AdaptiveAvgPool2d_ChannelsOut_t;
	typedef uint16_t AdaptiveAvgPool2d_predecessors_t;

// Dropout-Types
	typedef uint8_t Dropout_ID_t;
	typedef uint8_t Dropout_LayerNr_t;
	typedef uint8_t Dropout_predecessorNr_t;
	typedef uint32_t Dropout_DimensionInput_x_t;
	typedef uint32_t Dropout_DimensionOutput_x_t;
	typedef float Dropout_DropoutRate_t;
	typedef uint16_t Dropout_predecessors_t;

#endif