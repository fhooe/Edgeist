/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the rectified linear unit (ReLU) activation layer.
 */

#ifndef RELU_H
#define RELU_H

#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {

template <typename T>
class Model;

/**
 * @brief Rectified Linear Unit (ReLU) Activation Layer.
 *
 * Applies the function `f(x) = max(0, x)` element-wise.
 * Adds non-linearity to the network and prevents vanishing gradients.
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class Relu : public Layer<T> {
public:
    // CTor: Init the ReLU-Layer with pointers to config data and the chosen optimizer
    Relu(Model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {

        this->mHeader = static_cast<Neural_Network_ReLU_t*>(mPtrLayer);
        this->mInputData = nullptr;

        loadFromFlash();
    }

    // DTor: Frees dynamically allocated memory
    ~Relu() override
    {
        if (this->mInputData != nullptr) {
            delete[] this->mInputData;
            this->mInputData = nullptr;
        }
    }

    // executes forward pass and writes the result to the output
    auto forwardPass(const T* input_data, T* output_data, bool trainingFlag) -> ErrorType override
    {
        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        if (trainingFlag) {

            if (this->mIsLoaded != true) {
                return ErrorType::LayerNotInitialized;
            }
            // get memory for training
            if (this->mInputData == nullptr) {
                this->mInputData = new T[this->mHeader->dimensioninput_x];
            }

            if (this->mInputData != nullptr) {
                for (size_t i = 0; i < this->mHeader->dimensionoutput_x; i++) {
                    this->mInputData[i] = input_data[i];
                }
            }
        }

        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; i++) {
            output_data[i] = (input_data[i] > T(0)) ? input_data[i] : T(0);
        }

        return ErrorType::ok;
    }

    // executes the backward pass and calculates the gradient for the layer before
    auto backwardPass(const T* input_data, T* output_data) -> ErrorType override
    {

        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        if (this->mIsLoaded != true) {
            return ErrorType::LayerNotInitialized;
        }

        if (this->mInputData == nullptr) {
            return ErrorType::UnknownError;
        }

        for (uint32_t i = 0; i < this->mHeader->dimensioninput_x; i++) {
            output_data[i] = (this->mInputData[i] > T(0)) ? input_data[i] : T(0);
        }

        if (this->mInputData != nullptr) {
            delete[] this->mInputData;
            this->mInputData = nullptr;
        }

        return ErrorType::UnknownError;
    }

    // loads the training data from flash to SRAM
    auto loadFromFlash() -> ErrorType override
    {

        this->mIsLoaded = true;
        return ErrorType::ok;
    }

    // saves the trained values from SRAM to flash
    auto storeToFlash() -> ErrorType override
    {
        this->mIsLoaded = false;
        // Not implemented in this version, will only become relevant on the uController
        return ErrorType::UnknownError;
    }

    auto getOutputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensionoutput_x);
    }

    auto getInputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensioninput_x);
    }

    [[nodiscard]] auto header() const -> const Neural_Network_ReLU_t&
    {
        return *mHeader;
    }

private:
    void* mPtrLayer;

    void* mPtrData;

    // pointer to the layer in flash
    Neural_Network_ReLU_t* mHeader;

    // chosen optimizer
    OptimizerID mOptimizerType;
};
} // namespace Edgeist

#endif // RELU_H
