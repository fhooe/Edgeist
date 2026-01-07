/**
 * @file
 * @author David Muttenthaler
 * @brief Implements helper methods to load MNIST data
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
struct MNISTImage {
    float data[784];
    uint8_t label;
};

auto loadMNISTBatch(const std::string& filename) -> std::vector<MNISTImage>;
} // namespace Edgeist

#endif // MNIST_LOADER_H
