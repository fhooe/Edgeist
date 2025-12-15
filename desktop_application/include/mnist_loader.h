/**
 * @file
 * @author David Muttenthaler
 * @brief Implements helper methods to load MNIST data.
 */

#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <cstdint>
#include <string>
#include <vector>

namespace Edgeist {
/**
 * @brief Helper methods to load MNIST Data
 */
struct MnistImage {
    float data[784];
    uint8_t label;
};

std::vector<MnistImage> load_mnist_batch(const std::string& filename);
} // namespace Edgeist

#endif // MNIST_LOADER_H
