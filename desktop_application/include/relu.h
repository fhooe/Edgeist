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
    Relu(Model<T>* model, void* headerPointer, void* dataPointer, const OptimizerID optimizerType)
        : Layer<T>(model)
        , mPtrLayer(headerPointer)
        , mPtrData(dataPointer)
        , mHeader(static_cast<Neural_Network_ReLU_t*>(mPtrLayer))
        , mOptimizerType(optimizerType)
    {
        this->m_inputData = nullptr;

        Relu::loadFromFlash();
    }

    // DTor: Frees dynamically allocated memory
    ~Relu() override
    {
        if (this->m_inputData != nullptr) {
            delete[] this->m_inputData;
            this->m_inputData = nullptr;
        }
    }

    // executes forward pass and writes the result to the output
    auto forwardPass(const T* inputData, T* outputData, const bool trainingFlag) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }

        if (trainingFlag) {

            if (!this->m_isLoaded) {
                return ErrorType::LayerNotInitialized;
            }
            // get memory for training
            if (this->m_inputData == nullptr) {
                this->m_inputData = new T[this->mHeader->dimensioninput_x];
            }

            if (this->m_inputData != nullptr) {
                for (size_t i = 0; i < this->mHeader->dimensionoutput_x; i++) {
                    this->m_inputData[i] = inputData[i];
                }
            }
        }

        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; i++) {
            outputData[i] = (inputData[i] > T(0)) ? inputData[i] : T(0);
        }

        return ErrorType::OK;
    }

    // executes the backward pass and calculates the gradient for the layer before
    auto backwardPass(const T* inputData, T* outputData) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }

        if (!this->m_isLoaded) {
            return ErrorType::LayerNotInitialized;
        }

        if (this->m_inputData == nullptr) {
            return ErrorType::UnknownError;
        }

        for (uint32_t i = 0; i < this->mHeader->dimensioninput_x; i++) {
            outputData[i] = (this->m_inputData[i] > T(0)) ? inputData[i] : T(0);
        }

        if (this->m_inputData != nullptr) {
            delete[] this->m_inputData;
            this->m_inputData = nullptr;
        }

        return ErrorType::UnknownError;
    }

    // loads the training data from flash to SRAM
    auto loadFromFlash() -> ErrorType override
    {
        this->m_isLoaded = true;
        return ErrorType::OK;
    }

    // saves the trained values from SRAM to flash
    auto storeToFlash() -> ErrorType override
    {
        this->m_isLoaded = false;
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
