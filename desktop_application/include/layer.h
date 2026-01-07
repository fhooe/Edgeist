/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the base class for all neural network layers
 */

#ifndef LAYER_H
#define LAYER_H

#include "nmcf_error_types.h"
#include "object.h"
#include <cstdint>

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

    virtual auto initGradients() -> ErrorType
    {
        return ErrorType::OK;
    }

    virtual auto deleteGradients() -> ErrorType
    {
        return ErrorType::OK;
    }

    // Update weights and biases
    virtual auto update(uint32_t /* batchsize */) -> ErrorType
    {
        return ErrorType::OK;
    }

    virtual auto loadFromFlash() -> ErrorType = 0;

    virtual auto storeToFlash() -> ErrorType = 0;

    virtual auto getOutputSize() -> uint32_t = 0;

    virtual auto getInputSize() -> uint32_t = 0;

protected:
    // Flag that represents if the Layer has been loaded
    bool m_isLoaded = false;

    // pointer to the model Object that
    Model<T>* m_model = nullptr;

    // pointer to Array with Input Data
    // Used for Layers like ReLU, softMax, Maxpool, ...
    // To have the Data for Backwardspass available
    T* m_inputData;
};
} // namespace Edgeist

#endif // LAYER_H
