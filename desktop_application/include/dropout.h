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
    Dropout(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
        , mRng(std::random_device {}())
    {
        this->mHeader = static_cast<Neural_Network_Dropout_t*>(mPtrLayer);
        loadFromFlash();
    }

    auto forwardPass(const T* input_data, T* output_data, bool trainingflag) -> ErrorType override
    {
        if (!input_data || !output_data)
            return ErrorType::UnknownError;
        if (!this->mIsLoaded)
            return ErrorType::LayerNotInitialized;

        const size_t size = mHeader->dimensioninput_x;

        if (trainingflag) {
            if (mDropoutMask.size() != size)
                return ErrorType::DropoutMaskMissing;

            for (size_t i = 0; i < size; ++i) {
                output_data[i] = input_data[i] * mDropoutMask[i];
            }
        } else {
            for (size_t i = 0; i < size; ++i) {
                output_data[i] = input_data[i]; // do not change in inference mode
            }
        }

        return ErrorType::ok;
    }

    auto backwardPass(const T* grad_output, T* grad_input) -> ErrorType override
    {
        if (!grad_output || !grad_input)
            return ErrorType::UnknownError;
        if (!this->mIsLoaded)
            return ErrorType::LayerNotInitialized;

        const size_t size = mHeader->dimensioninput_x;

        if (mDropoutMask.size() != size)
            return ErrorType::DropoutMaskMissing;

        for (size_t i = 0; i < size; ++i) {
            grad_input[i] = grad_output[i] * mDropoutMask[i];
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
