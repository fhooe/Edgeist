/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the dropout layer.
 */

#ifndef DROPOUT_H
#define DROPOUT_H

#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"
#include <random>
#include <vector>

namespace Edgeist {
/**
 * @brief Dropout layer.
 *
 * Randomly zeroes some of the elements of the input tensor during training
 * with a probability `p`, helping prevent overfitting.
 *
 * No effect during inference (pass-through).
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class Dropout : public Layer<T> {
public:
    Dropout(Model<T>* model, void* headerPointer, void* dataPointer, const OptimizerID optimizerType)
        : Layer<T>(model)
        , mPtrLayer(headerPointer)
        , mPtrData(dataPointer)
        , mHeader(static_cast<Neural_Network_Dropout_t*>(mPtrLayer))
        , mOptimizerType(optimizerType)
        , mRng(std::random_device {}())
    {
        Dropout::loadFromFlash();
    }

    auto forwardPass(const T* inputData, T* outputData, const bool trainingFlag) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }
        if (!this->mIsLoaded) {
            return ErrorType::LayerNotInitialized;
        }

        const size_t size = mHeader->dimensioninput_x;

        if (trainingFlag) {
            if (mDropoutMask.size() != size) {
                return ErrorType::DropoutMaskMissing;
            }

            for (size_t i = 0; i < size; ++i) {
                outputData[i] = inputData[i] * mDropoutMask[i];
            }
        } else {
            for (size_t i = 0; i < size; ++i) {
                outputData[i] = inputData[i]; // do not change in inference mode
            }
        }

        return ErrorType::ok;
    }

    auto backwardPass(const T* gradOutput, T* gradInput) -> ErrorType override
    {
        if (gradOutput == nullptr || gradInput == nullptr) {
            return ErrorType::UnknownError;
        }
        if (!this->mIsLoaded) {
            return ErrorType::LayerNotInitialized;
        }

        const size_t size = mHeader->dimensioninput_x;

        if (mDropoutMask.size() != size) {
            return ErrorType::DropoutMaskMissing;
        }

        for (size_t i = 0; i < size; ++i) {
            gradInput[i] = gradOutput[i] * mDropoutMask[i];
        }

        return ErrorType::ok;
    }

    auto initGradients() -> ErrorType override
    {
        const auto& H = *this->mHeader;
        const size_t size = size_t(H.dimensioninput_x);
        const float rate = H.dropoutrate;

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        mDropoutMask.resize(size);

        for (size_t i = 0; i < size; ++i) {
            bool keep = dist(mRng) >= rate;
            mDropoutMask[i] = keep ? T(1.0f) / (1.0f - rate) : T(0);
        }

        return ErrorType::ok;
    }

    auto deleteGradients() -> ErrorType override
    {
        mDropoutMask.clear();
        return ErrorType::ok;
    }

    auto loadFromFlash() -> ErrorType override
    {
        this->mIsLoaded = true;
        return ErrorType::ok;
    }

    auto storeToFlash() -> ErrorType override
    {
        // Not implemented
        return ErrorType::ok;
    }

    auto getOutputSize() -> uint32_t override
    {
        return mHeader->dimensionoutput_x;
    }

    auto getInputSize() -> uint32_t override
    {
        return mHeader->dimensioninput_x;
    }

private:
    void* mPtrLayer;
    void* mPtrData;
    Neural_Network_Dropout_t* mHeader;
    OptimizerID mOptimizerType;

    std::vector<T> mDropoutMask;
    std::mt19937 mRng;
};
} // namespace Edgeist

#endif // DROPOUT_H
