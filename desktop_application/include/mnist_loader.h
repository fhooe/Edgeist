/**
 * @file
 * @author David Muttenthaler
 * @brief Implements helper methods to load MNIST data
 */

#ifndef EDGEIST_MNIST_LOADER_H
#define EDGEIST_MNIST_LOADER_H

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

std::vector<MNISTImage> loadMNISTBatch(const std::string& filename);
} // namespace Edgeist

#endif // EDGEIST_MNIST_LOADER_H
