/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the 2D max pooling layer.
 */

#ifndef MAX_POOL_2D_H
#define MAX_POOL_2D_H

#include "nmcf_error_types.h"
#include "optimizer_data_types.h"
#include <limits>

namespace Edgeist {
template <typename T>
class Model;

/**
 * @brief 2D max pooling layer.
 *
 * Applies a 2D max pooling operation over the input, reducing spatial
 * dimensions by taking the maximum value in each kernel region.
 *
 * Helps with spatial invariance and downsampling.
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class MaxPool2D : public Layer<T> {
public:
    MaxPool2D(Model<T>* model, void* headerPointer, void* dataPointer, const OptimizerID optimizerType)
        : Layer<T>(model)
        , mPtrLayer(headerPointer)
        , mPtrData(dataPointer)
        , mHeader(static_cast<Neural_Network_MaxPool2d_t*>(mPtrLayer))
        , mOptimizerType(optimizerType)
    {
        MaxPool2D::loadFromFlash();
        mArgmax = nullptr;
    }

    ~MaxPool2D() override
    {
        if (mArgmax != nullptr) {
            delete[] mArgmax;
            mArgmax = nullptr;
        }
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
        const int kH = H.kernelsize;
        const int kW = H.kernelsize;
        const int padH = H.padding;
        const int padW = H.padding;
        const int strideH = H.stride;
        const int strideW = H.stride;

        // allocate argmax storage
        size_t outSize = size_t(C) * H_out * W_out;
        delete[] mArgmax;
        mArgmax = new (std::nothrow) uint32_t[outSize];
        if (mArgmax == nullptr) {
            return ErrorType::UnknownError;
        }

        // iterate over channels and spatial dims
        for (int c = 0; c < C; ++c) {
            for (int oy = 0; oy < H_out; ++oy) {
                for (int ox = 0; ox < W_out; ++ox) {
                    T maxVal = std::numeric_limits<T>::lowest();
                    uint32_t maxIdx = 0;
                    // window
                    for (int ky = 0; ky < kH; ++ky) {
                        for (int kx = 0; kx < kW; ++kx) {
                            int inY = oy * strideH + ky - padH;
                            int inX = ox * strideW + kx - padW;
                            if (inY < 0 || inY >= H_in || inX < 0 || inX >= W_in)
                                continue;
                            size_t idx = size_t(c) * H_in * W_in + inY * W_in + inX;
                            T val = inputData[idx];
                            if (val > maxVal) {
                                maxVal = val;
                                maxIdx = idx;
                            }
                        }
                    }
                    size_t outIdx = size_t(c) * H_out * W_out + oy * W_out + ox;
                    outputData[outIdx] = maxVal;
                    mArgmax[outIdx] = maxIdx;
                }
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
        if (mArgmax == nullptr) {
            return ErrorType::UnknownError;
        }

        const auto& H = *this->mHeader;
        const int C = H.channelsin;
        const int H_in = H.dimensioninput_y;
        const int W_in = H.dimensioninput_x;
        const int H_out = H.dimensionoutput_y;
        const int W_out = H.dimensionoutput_x;

        // zero initialize grad_input
        size_t inSize = size_t(C) * H_in * W_in;
        for (size_t i = 0; i < inSize; ++i) {
            gradInput[i] = T(0);
        }

        // propagate gradients
        for (size_t outIdx = 0; outIdx < size_t(C) * H_out * W_out; ++outIdx) {
            uint32_t inIdx = mArgmax[outIdx];
            gradInput[inIdx] += gradOutput[outIdx];
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
        // not implemented for this version
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
    Neural_Network_MaxPool2d_t* mHeader;
    OptimizerID mOptimizerType;
    uint32_t* mArgmax;
};
} // namespace Edgeist

#endif // MAX_POOL_2D_H
