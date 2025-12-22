#ifndef MODELENUMS_H
#define MODELENUMS_H

enum class DataEncodingIDs
{
	 float32_ID = 0,
	 float16_ID = 1,
	 float64_ID = 2,
	 int32_ID = 3,
	 int64_ID = 4,
};

enum class LayerIDs
{
	 AdaptiveAvgPool1d_ID = 8,
	 AdaptiveAvgPool2d_ID = 9,
	 BatchNorm1d_ID = 10,
	 BatchNorm2d_ID = 11,
	 Conv2d_ID = 3,
	 Dropout_ID = 4,
	 Flatten_ID = 2,
	 Linear_ID = 1,
	 MaxPool2d_ID = 5,
	 ReLU_ID = 6,
	 Softmax_ID = 7,
};

#endif