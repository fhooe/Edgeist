#pragma once
#include "Modeltypes.h"
#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"
#include <cmath>
#include <limits>
#include <vector>

/**
 * @author David Muttenthaler
 * @date 25-06-2025
 *
 * @brief 2D Adaptive Average Pooling Layer.
 *
 * Performs average pooling on 2D input (e.g. images), adapting the pooling
 * regions to achieve a desired fixed output size regardless of input size.
 *
 * Useful for global or spatial pooling before fully connected layers.
 */
template <typename T>
class AdaptiveAvgPool2d : public Layer<T> {
public:
    AdaptiveAvgPool2d(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {
        this->mHeader = static_cast<Neural_Network_AdaptiveAvgPool2d_t*>(mPtrLayer);
        loadFromFlash();
    }

    ErrorType forwardPass(const T* input_data, T* output_data, bool /* trainingflag */) override
    {
        if (!input_data || !output_data)
            return ErrorType::UnknownError;
        if (!this->mIsLoaded)
            return ErrorType::LayerNotInitialized;

        const auto& H = *this->mHeader;
        const int C = H.channelsin;
        const int H_in = H.dimensioninput_y;
        const int W_in = H.dimensioninput_x;
        const int H_out = H.dimensionoutput_y;
        const int W_out = H.dimensionoutput_x;

        for (int c = 0; c < C; ++c) {
            for (int oy = 0; oy < H_out; ++oy) {
                // Adaptive Start/End f�r y
                int y_start = std::floor(oy * H_in / static_cast<float>(H_out));
                int y_end = std::ceil((oy + 1) * H_in / static_cast<float>(H_out));

                for (int ox = 0; ox < W_out; ++ox) {
                    // Adaptive Start/End f�r x
                    int x_start = std::floor(ox * W_in / static_cast<float>(W_out));
                    int x_end = std::ceil((ox + 1) * W_in / static_cast<float>(W_out));

                    T sum = 0;
                    int count = 0;
                    for (int iy = y_start; iy < y_end; ++iy) {
                        for (int ix = x_start; ix < x_end; ++ix) {
                            size_t in_idx = size_t(c) * H_in * W_in + iy * W_in + ix;
                            sum += input_data[in_idx];
                            ++count;
                        }
                    }

                    size_t out_idx = size_t(c) * H_out * W_out + oy * W_out + ox;
                    output_data[out_idx] = count > 0 ? sum / static_cast<T>(count) : T(0);
                }
            }
        }

        return ErrorType::ok;
    }

    ErrorType backwardPass(const T* grad_output, T* grad_input) override
    {
        if (!grad_output || !grad_input)
            return ErrorType::UnknownError;
        if (!this->mIsLoaded)
            return ErrorType::LayerNotInitialized;

        const auto& H = *this->mHeader;
        const int C = H.channelsin;
        const int H_in = H.dimensioninput_y;
        const int W_in = H.dimensioninput_x;
        const int H_out = H.dimensionoutput_y;
        const int W_out = H.dimensionoutput_x;

        // Initialisiere grad_input mit 0
        size_t inSize = size_t(C) * H_in * W_in;
        for (size_t i = 0; i < inSize; ++i) {
            grad_input[i] = T(0);
        }

        // Gradienten verteilen
        for (int c = 0; c < C; ++c) {
            for (int oy = 0; oy < H_out; ++oy) {
                int y_start = std::floor(oy * H_in / static_cast<float>(H_out));
                int y_end = std::ceil((oy + 1) * H_in / static_cast<float>(H_out));

                for (int ox = 0; ox < W_out; ++ox) {
                    int x_start = std::floor(ox * W_in / static_cast<float>(W_out));
                    int x_end = std::ceil((ox + 1) * W_in / static_cast<float>(W_out));

                    size_t out_idx = size_t(c) * H_out * W_out + oy * W_out + ox;
                    T grad = grad_output[out_idx];
                    int count = (y_end - y_start) * (x_end - x_start);
                    T grad_val = count > 0 ? grad / static_cast<T>(count) : T(0);

                    for (int iy = y_start; iy < y_end; ++iy) {
                        for (int ix = x_start; ix < x_end; ++ix) {
                            size_t in_idx = size_t(c) * H_in * W_in + iy * W_in + ix;
                            grad_input[in_idx] += grad_val;
                        }
                    }
                }
            }
        }

        return ErrorType::ok;
    }

    ErrorType loadFromFlash() override
    {
        this->mIsLoaded = true;
        return ErrorType::ok;
    }

    ErrorType storeToFlash() override
    {
        // not implemented
        return ErrorType::ok;
    }

    uint32_t getOutputSize() override
    {
        return this->mHeader->channelsin * this->mHeader->dimensionoutput_x * this->mHeader->dimensionoutput_y;
    }

    uint32_t getInputSize() override
    {
        return this->mHeader->channelsin * this->mHeader->dimensioninput_x * this->mHeader->dimensioninput_y;
    }

private:
    void* mPtrLayer;
    void* mPtrData;
    Neural_Network_AdaptiveAvgPool2d_t* mHeader;
    OptimizerID mOptimizerType;
};
