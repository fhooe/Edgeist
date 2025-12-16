/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the 2D adaptive average pooling layer.
 */

#ifndef ADAPTIVE_AVG_POOL_2D_H
#define ADAPTIVE_AVG_POOL_2D_H

#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"
#include <cmath>

namespace Edgeist {
/**
 * @brief 2D adaptive average pooling layer.
 *
 * Performs average pooling on 2D input (e.g. images), adapting the pooling
 * regions to achieve a desired fixed output size regardless of input size.
 *
 * Useful for global or spatial pooling before fully connected layers.
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class AdaptiveAvgPool2D : public Layer<T> {
public:
    AdaptiveAvgPool2D(Model<T>* model, void* headerPointer, void* dataPointer, const OptimizerID optimizerType)
        : Layer<T>(model)
        , mPtrLayer(headerPointer)
        , mPtrData(dataPointer)
        , mHeader(static_cast<Neural_Network_AdaptiveAvgPool2d_t*>(mPtrLayer))
        , mOptimizerType(optimizerType)
    {
        AdaptiveAvgPool2D::loadFromFlash();
    }

    auto forwardPass(const T* inputData, T* outputData, bool /* trainingflag */) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }
        if (!this->mIsLoaded) {
            return ErrorType::LayerNotInitialized;
        }

        const auto& H = *this->mHeader;
        const int C = H.channelsin;
        const int H_in = H.dimensioninput_y;
        const int W_in = H.dimensioninput_x;
        const int H_out = H.dimensionoutput_y;
        const int W_out = H.dimensionoutput_x;

        for (int c = 0; c < C; ++c) {
            for (int oy = 0; oy < H_out; ++oy) {
                // adaptive start/end for y
                int y_start = std::floor(oy * H_in / static_cast<float>(H_out));
                int y_end = std::ceil((oy + 1) * H_in / static_cast<float>(H_out));

                for (int ox = 0; ox < W_out; ++ox) {
                    // adaptive start/end for x
                    int x_start = std::floor(ox * W_in / static_cast<float>(W_out));
                    int x_end = std::ceil((ox + 1) * W_in / static_cast<float>(W_out));

                    T sum = 0;
                    int count = 0;
                    for (int iy = y_start; iy < y_end; ++iy) {
                        for (int ix = x_start; ix < x_end; ++ix) {
                            size_t in_idx = size_t(c) * H_in * W_in + iy * W_in + ix;
                            sum += inputData[in_idx];
                            ++count;
                        }
                    }

                    size_t out_idx = size_t(c) * H_out * W_out + oy * W_out + ox;
                    outputData[out_idx] = count > 0 ? sum / static_cast<T>(count) : T(0);
                }
            }
        }

        return ErrorType::ok;
    }

    auto backwardPass(const T* gradOutput, T* gradInput) -> ErrorType override
    {
        if ((gradOutput == nullptr) || (gradInput == nullptr)) {
            return ErrorType::UnknownError;
        }
        if (!this->mIsLoaded) {
            return ErrorType::LayerNotInitialized;
        }

        const auto& H = *this->mHeader;
        const int C = H.channelsin;
        const int H_in = H.dimensioninput_y;
        const int W_in = H.dimensioninput_x;
        const int H_out = H.dimensionoutput_y;
        const int W_out = H.dimensionoutput_x;

        // init grad_input with 0
        size_t inSize = size_t(C) * H_in * W_in;
        for (size_t i = 0; i < inSize; ++i) {
            gradInput[i] = T(0);
        }

        // distribute gradient
        for (int c = 0; c < C; ++c) {
            for (int oy = 0; oy < H_out; ++oy) {
                int y_start = std::floor(oy * H_in / static_cast<float>(H_out));
                int y_end = std::ceil((oy + 1) * H_in / static_cast<float>(H_out));

                for (int ox = 0; ox < W_out; ++ox) {
                    int x_start = std::floor(ox * W_in / static_cast<float>(W_out));
                    int x_end = std::ceil((ox + 1) * W_in / static_cast<float>(W_out));

                    size_t out_idx = size_t(c) * H_out * W_out + oy * W_out + ox;
                    T grad = gradOutput[out_idx];
                    int count = (y_end - y_start) * (x_end - x_start);
                    T grad_val = count > 0 ? grad / static_cast<T>(count) : T(0);

                    for (int iy = y_start; iy < y_end; ++iy) {
                        for (int ix = x_start; ix < x_end; ++ix) {
                            size_t in_idx = size_t(c) * H_in * W_in + iy * W_in + ix;
                            gradInput[in_idx] += grad_val;
                        }
                    }
                }
            }
        }

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

    auto getOutputSize() -> uint32_t override
    {
        return this->mHeader->channelsin * this->mHeader->dimensionoutput_x * this->mHeader->dimensionoutput_y;
    }

    auto getInputSize() -> uint32_t override
    {
        return this->mHeader->channelsin * this->mHeader->dimensioninput_x * this->mHeader->dimensioninput_y;
    }

private:
    void* mPtrLayer;
    void* mPtrData;
    Neural_Network_AdaptiveAvgPool2d_t* mHeader;
    OptimizerID mOptimizerType;
};
} // namespace Edgeist
#endif // ADAPTIVE_AVG_POOL_2D_H
