/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the softmax activation layer.
 */

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"
#include <Modelstructs.h>
#include <cmath>

namespace Edgeist {
template <typename T>
class Model;

/**
 * @brief Softmax activation layer.
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
    Softmax(Model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {

        this->mHeader = static_cast<Neural_Network_Softmax_t*>(mPtrLayer);
        this->mInputData = nullptr;

        loadFromFlash();
    }

    // DTor: Frees dynamically allocated memory
    ~Softmax() override
    {

        if (this->mInputData != nullptr) {
            delete[] this->mInputData;
            this->mInputData = nullptr;
        }
    }

    // Performs the forward pass; calculates softmax output from input data
    // input_data: output of the previous layer
    // output_Data: output of this layer
    auto forwardPass(const T* input_data, T* output_data, bool trainingflag) -> ErrorType override
    {
        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        if (this->mIsLoaded != true) {
            return ErrorType::LayerNotInitialized;
        }

        // Find the maximum value of the input for numerical stabilization of the exponential function.
        T maxVal = input_data[0];
        for (size_t i = 1; i < this->mHeader->dimensioninput_x; i++) {
            if (input_data[i] > maxVal) {
                maxVal = input_data[i];
            }
        }

        // Calculate exponential values of the inputs shifted by maxVal and sum them up.
        T sumExponents = T(0);
        T* mPtrExponents = new T[this->mHeader->dimensioninput_x];

        for (size_t i = 0; i < this->mHeader->dimensioninput_x; i++) {
            mPtrExponents[i] = std::exp(input_data[i] - maxVal);
            sumExponents += mPtrExponents[i];
        }

        if (trainingflag) {
            // Allocate memory for temporary storage of input (for backpropagation)
            if (this->mInputData != nullptr) {
                delete[] this->mInputData;
            }
            this->mInputData = new T[this->mHeader->dimensioninput_x];

            // Normalize exponential values for Softmax output
            if (this->mInputData != nullptr) {
                for (size_t i = 0; i < this->mHeader->dimensionoutput_x; i++) {
                    this->mInputData[i] = input_data[i];
                }
            }
        }
        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; i++) {
            output_data[i] = mPtrExponents[i] / sumExponents;
        }

        delete[] mPtrExponents;
        mPtrExponents = nullptr;

        return ErrorType::ok;
    }

    // Performs the backward pass; calculates the error gradient of the softmax layer
    // input_data: output from forwardPass
    // output_Data: is gradient
    auto backwardPass(const T* input_data, T* output_data) -> ErrorType override
    {
        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        if (this->mIsLoaded != true) {
            return ErrorType::LayerNotInitialized;
        }

        // Softmax + cross-entropy derivative: p - y
        for (size_t i = 0; i < this->mHeader->dimensioninput_x; i++) {
            output_data[i] = input_data[i] - this->mModel->mPtrExpectedOutputData[i];
        }

        return ErrorType::ok;
    }

    // Loads the trainable values from Flash into SRAM
    auto loadFromFlash() -> ErrorType override
    {

        this->mIsLoaded = true;
        // Not relevant for ReLU, as there are no weights and biases
        return ErrorType::ok;
    }

    // Stores the trainable values from SRAM in flash memory
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

    [[nodiscard]] auto header() const -> const Neural_Network_Softmax_t&
    {
        return *mHeader;
    }

private:
    void* mPtrLayer;

    void* mPtrData;

    // Pointer to the Layer in Flash
    Neural_Network_Softmax_t* mHeader;

    // chosen optimizer
    OptimizerID mOptimizerType;
};
} // namespace Edgeist

#endif // SOFTMAX_H
