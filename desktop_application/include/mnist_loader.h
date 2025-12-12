#pragma once
#include <cstdint>
#include <string>
#include <vector>

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
