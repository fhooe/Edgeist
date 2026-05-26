/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the base class for all neural network layers
 */

#ifndef EDGEIST_LAYER_H
#define EDGEIST_LAYER_H

#include <cstdint>

#include "nmcf_error_types.h"
#include "object.h"

namespace Edgeist {
template <typename T>
class Model;

/**
 * @brief Base class for all neural network layers
 *
 * This abstract class defines the common interface and behavior for all neural
 * network layers, including forward and backward passes, weight updates,...
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class Layer : public Object {
public:
    explicit Layer(Model<T>* model)
        : m_model(model)
    {
    }

    ~Layer() override = default;

    // executes forward pass and writes the result to the output
    virtual auto forwardPass(const T* inputData, T* outputData, bool trainingFlag) -> ErrorType = 0;

    // executes the backward pass and calculates the gradient for the layer before
    virtual auto backwardPass(const T* inputData, T* outputData) -> ErrorType = 0;

    virtual ErrorType initGradients()
    {
        return ErrorType::OK;
    }

    virtual ErrorType deleteGradients()
    {
        return ErrorType::OK;
    }

    // Update weights and biases
    virtual ErrorType update(uint32_t /* batchsize */)
    {
        return ErrorType::OK;
    }

    virtual ErrorType loadFromFlash() = 0;

    virtual ErrorType storeToFlash() = 0;

    virtual uint32_t getOutputSize() = 0;

    virtual uint32_t getInputSize() = 0;

protected:
    // Flag that represents if the Layer has been loaded
    bool m_isLoaded = false;

    // pointer to the model Object that
    Model<T>* m_model = nullptr;

    // pointer to Array with Input Data
    // Used for Layers like ReLU, softMax, Maxpool, ...
    // To have the Data for Backwardspass available
    T* m_inputData = nullptr;
};
} // namespace Edgeist

#endif // EDGEIST_LAYER_H
