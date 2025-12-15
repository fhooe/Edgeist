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
class model;

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
    Flatten(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {

        this->mHeader = static_cast<Neural_Network_Flatten_t*>(mPtrLayer);
    }

    // DTor: frees dynamically allocated memory
    ~Flatten() override = default;

    // executes forward pass and writes the result to the output
    auto forwardPass(const T* input_data, T* output_data, bool /* trainingflag */) -> ErrorType override
    {
        for (size_t i = 0; i < mHeader->dimensioninput_x; i++) {
            output_data[i] = input_data[i];
        }
        return ErrorType::ok;
    }

    // executes the backward pass and calculates the gradient for the layer before
    auto backwardPass(const T* input_data, T* output_data) -> ErrorType override
    {
        for (size_t i = 0; i < mHeader->dimensioninput_x; i++) {
            output_data[i] = input_data[i];
        }
        return ErrorType::ok;
    }

    // loads the training data from flash to SRAM
    auto loadFromFlash() -> ErrorType override
    {
        ;
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
