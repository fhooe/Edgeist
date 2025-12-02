#pragma once
#include <vector>
#include <string>
#include <cstdint>

/**
 * @author David Muttenthaler
 * @date 25-06-2025
 *
 * @brief Helper methode to load Mnist Data
 *
 */

struct MnistImage {
	float data[784];
	uint8_t label;
};

std::vector<MnistImage> load_mnist_batch(const std::string& filename);
