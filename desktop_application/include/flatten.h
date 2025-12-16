/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the flatten layer.
 */

#ifndef FLATTEN_H
#define FLATTEN_H

#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {
template <typename T>
class Model;

/**
 * @brief Flatten layer.
 *
 * Flattens a multi-dimensional input tensor into a 1D tensor.
 * Commonly used to connect convolutional layers with fully connected layers.
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class Flatten : public Layer<T> {
public:
    // CTor: Init the ReLU-Layer with pointers to config data and the chosen optimizer
    Flatten(Model<T>* model, void* headerPointer, void* dataPointer, const OptimizerID optimizerType)
        : Layer<T>(model)
        , mPtrLayer(headerPointer)
        , mPtrData(dataPointer)
        , mHeader(static_cast<Neural_Network_Flatten_t*>(mPtrLayer))
        , mOptimizerType(optimizerType)
    {
    }

    // DTor: frees dynamically allocated memory
    ~Flatten() override = default;

    // executes forward pass and writes the result to the output
    auto forwardPass(const T* inputData, T* outputData, bool /* trainingflag */) -> ErrorType override
    {
        for (size_t i = 0; i < mHeader->dimensioninput_x; i++) {
            outputData[i] = inputData[i];
        }
        return ErrorType::ok;
    }

    // executes the backward pass and calculates the gradient for the layer before
    auto backwardPass(const T* inputData, T* outputData) -> ErrorType override
    {
        for (size_t i = 0; i < mHeader->dimensioninput_x; i++) {
            outputData[i] = inputData[i];
        }
        return ErrorType::ok;
    }

    // loads the training data from flash to SRAM
    auto loadFromFlash() -> ErrorType override
    {
        return ErrorType::ok;
    }

    // saves the trained values from SRAM to flash
    auto storeToFlash() -> ErrorType override
    {
        return ErrorType::ok;
    }

    auto getOutputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensionoutput_x);
    }

    auto getInputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensioninput_x);
    }

    [[nodiscard]] auto header() const -> const Neural_Network_Flatten_t&
    {
        return *mHeader;
    }

private:
    void* mPtrLayer;

    void* mPtrData;

    // Pointer to the Layer in Flash
    Neural_Network_Flatten_t* mHeader;

    // chosen optimizer
    OptimizerID mOptimizerType;
};
} // namespace Edgeist

#endif // FLATTEN_H
