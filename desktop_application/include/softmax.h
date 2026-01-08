/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the softmax activation layer
 */

#ifndef EDGEIST_SOFTMAX_H
#define EDGEIST_SOFTMAX_H

#include <cmath>

#include "layer.h"
#include "model_structs.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {
template <typename T>
class Model;

/**
 * @brief Softmax activation layer
 *
 * Converts raw logits into probabilities by exponentiating and normalizing.
 * Used as the final layer in multi-class classification tasks.
 *
 * Output is a probability distribution (sums to 1).
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class Softmax : public Layer<T> {
public:
    // CTor: Init the Softmax-Layer with pointers to config data and the chosen optimizer
    Softmax(Model<T>* model, void* headerPointer, void* dataPointer, const OptimizerID optimizerType)
        : Layer<T>(model)
        , mPtrLayer(headerPointer)
        , mPtrData(dataPointer)
        , mHeader(static_cast<Nmcm::Softmax::NeuralNetwork*>(mPtrLayer))
        , mOptimizerType(optimizerType)
    {
        this->m_inputData = nullptr;

        Softmax::loadFromFlash();
    }

    // DTor: Frees dynamically allocated memory
    ~Softmax() override
    {
        if (this->m_inputData != nullptr) {
            delete[] this->m_inputData;
            this->m_inputData = nullptr;
        }
    }

    // Performs the forward pass; calculates softmax output from input data
    // input_data: output of the previous layer
    // output_Data: output of this layer
    auto forwardPass(const T* inputData, T* outputData, const bool trainingFlag) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }

        if (!this->m_isLoaded) {
            return ErrorType::LayerNotInitialized;
        }

        // Find the maximum value of the input for numerical stabilization of the exponential function.
        T maxVal = inputData[0];
        for (size_t i = 1; i < this->mHeader->dimensionInputX; i++) {
            if (inputData[i] > maxVal) {
                maxVal = inputData[i];
            }
        }

        // Calculate exponential values of the inputs shifted by maxVal and sum them up.
        T sumExponents = T(0);
        T* mPtrExponents = new T[this->mHeader->dimensionInputX];

        for (size_t i = 0; i < this->mHeader->dimensionInputX; i++) {
            mPtrExponents[i] = std::exp(inputData[i] - maxVal);
            sumExponents += mPtrExponents[i];
        }

        if (trainingFlag) {
            // Allocate memory for temporary storage of input (for backpropagation)
            if (this->m_inputData != nullptr) {
                delete[] this->m_inputData;
            }
            this->m_inputData = new T[this->mHeader->dimensionInputX];

            // Normalize exponential values for Softmax output
            if (this->m_inputData != nullptr) {
                for (size_t i = 0; i < this->mHeader->dimensionOutputX; i++) {
                    this->m_inputData[i] = inputData[i];
                }
            }
        }
        for (size_t i = 0; i < this->mHeader->dimensionOutputX; i++) {
            outputData[i] = mPtrExponents[i] / sumExponents;
        }

        delete[] mPtrExponents;
        mPtrExponents = nullptr;

        return ErrorType::OK;
    }

    // Performs the backward pass; calculates the error gradient of the softmax layer
    // input_data: output from forwardPass
    // output_Data: is gradient
    auto backwardPass(const T* inputData, T* outputData) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }

        if (!this->m_isLoaded) {
            return ErrorType::LayerNotInitialized;
        }

        // Softmax + cross-entropy derivative: p - y
        for (size_t i = 0; i < this->mHeader->dimensionInputX; i++) {
            outputData[i] = inputData[i] - this->m_model->mPtrExpectedOutputData[i];
        }

        return ErrorType::OK;
    }

    // Loads the trainable values from Flash into SRAM
    auto loadFromFlash() -> ErrorType override
    {
        this->m_isLoaded = true;
        // Not relevant for ReLU, as there are no weights and biases
        return ErrorType::OK;
    }

    // Stores the trainable values from SRAM in flash memory
    auto storeToFlash() -> ErrorType override
    {
        this->m_isLoaded = false;
        // Not implemented in this version, will only become relevant on the uController
        return ErrorType::UnknownError;
    }

    auto getOutputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensionOutputX);
    }

    auto getInputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensionInputX);
    }

    [[nodiscard]] auto header() const -> const Nmcm::Softmax::NeuralNetwork&
    {
        return *mHeader;
    }

private:
    void* mPtrLayer;

    void* mPtrData;

    // Pointer to the Layer in Flash
    Nmcm::Softmax::NeuralNetwork* mHeader;

    // chosen optimizer
    OptimizerID mOptimizerType;
};
} // namespace Edgeist

#endif // EDGEIST_SOFTMAX_H
