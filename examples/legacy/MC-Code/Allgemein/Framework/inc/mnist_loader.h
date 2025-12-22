#pragma once
#include <vector>
#include <string>
#include <cstdint>

struct MnistImage {
	float data[784];
	uint8_t label;
	
	MnistImage() = default;
	
	MnistImage(const uint32_t* inputData, uint8_t inputLabel) : label(inputLabel) 
	{
		for (uint32_t i = 0; i < 784; i++)
		{
			data[i] = (float)(inputData[i]);
		}
		//std::copy(inputData, &inputData[784], (uint32_t*)data);
	}
};

std::vector<MnistImage> load_mnist_batch(const std::string& filename);

void load_mnist_small(std::vector<MnistImage>& images);
