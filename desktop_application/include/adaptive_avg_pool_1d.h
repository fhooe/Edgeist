/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the 1D adaptive average pooling layer.
 */

#ifndef ADAPTIVE_AVG_POOL_1D_H
#define ADAPTIVE_AVG_POOL_1D_H

#include "Modelstructs.h"
#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"
#include <cmath>

namespace Edgeist {
/**
 * @brief 1D adaptive average pooling layer.
 *
 * This layer applies average pooling over a 1D input signal, automatically
 * adjusting the kernel and stride size to produce the specified output size.
 *
 * Commonly used to reduce variable-length input to a fixed-size representation.
 *
 * Example use: time series or sequence feature pooling.
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class AdaptiveAvgPool1d : public Layer<T> {
public:
    AdaptiveAvgPool1d(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {
        this->mHeader = static_cast<Neural_Network_AdaptiveAvgPool1d_t*>(mPtrLayer);
        loadFromFlash();
    }

    auto forwardPass(const T* input_data, T* output_data, bool /* trainingflag */) -> ErrorType override
    {
        if (!input_data || !output_data)
            return ErrorType::UnknownError;
        if (!this->mIsLoaded)
            return ErrorType::LayerNotInitialized;

        const auto& H = *this->mHeader;
        const int C = H.channelsin;
        const int W_in = H.dimensioninput_x;
        const int W_out = H.dimensionoutput_x;

        for (int c = 0; c < C; ++c) {
            for (int ox = 0; ox < W_out; ++ox) {
                int x_start = std::floor(ox * W_in / static_cast<float>(W_out));
                int x_end = std::ceil((ox + 1) * W_in / static_cast<float>(W_out));

                T sum = 0;
                int count = 0;
                for (int ix = x_start; ix < x_end; ++ix) {
                    size_t in_idx = size_t(c) * W_in + ix;
                    sum += input_data[in_idx];
                    ++count;
                }

                size_t out_idx = size_t(c) * W_out + ox;
                output_data[out_idx] = count > 0 ? sum / static_cast<T>(count) : T(0);
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

        const auto& H = *this->mHeader;
        const int C = H.channelsin;
        const int W_in = H.dimensioninput_x;
        const int W_out = H.dimensionoutput_x;

        size_t inSize = size_t(C) * W_in;
        for (size_t i = 0; i < inSize; ++i) {
            grad_input[i] = T(0);
        }

        for (int c = 0; c < C; ++c) {
            for (int ox = 0; ox < W_out; ++ox) {
                int x_start = std::floor(ox * W_in / static_cast<float>(W_out));
                int x_end = std::ceil((ox + 1) * W_in / static_cast<float>(W_out));

                size_t out_idx = size_t(c) * W_out + ox;
                T grad = grad_output[out_idx];
                int count = x_end - x_start;
                T grad_val = count > 0 ? grad / static_cast<T>(count) : T(0);

                for (int ix = x_start; ix < x_end; ++ix) {
                    size_t in_idx = size_t(c) * W_in + ix;
                    grad_input[in_idx] += grad_val;
                }
            }
        }

        return ErrorType::ok;
    }

    // init dropout mask for mini batch
    auto initGradients() -> ErrorType override
    {
        // TODO
        return ErrorType::ok;
    }

    // delete dropout mask
    auto deleteGradients() -> ErrorType override
    {
        // TODO
        return ErrorType::ok;
    }

    auto loadFromFlash() -> ErrorType override
    {
        this->mIsLoaded = true;
        return ErrorType::ok;
    }

    auto storeToFlash() -> ErrorType override
    {
        // not implemented
        return ErrorType::ok;
    }

    auto getOutputSize() -> uint32_t override { return this->mHeader->channelsin * this->mHeader->dimensionoutput_x; }

    auto getInputSize() -> uint32_t override { return this->mHeader->channelsin * this->mHeader->dimensioninput_x; }

private:
    void* mPtrLayer;
    void* mPtrData;
    Neural_Network_AdaptiveAvgPool1d_t* mHeader;
    OptimizerID mOptimizerType;
};
} // namespace Edgeist

#endif // ADAPTIVE_AVG_POOL_1D_H
