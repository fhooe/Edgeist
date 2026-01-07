/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the 2D max pooling layer
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
 * @brief 2D max pooling layer
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
        , m_ptrLayer(headerPointer)
        , m_ptrData(dataPointer)
        , m_header(static_cast<Nmcm::MaxPool2d::NeuralNetwork_t*>(m_ptrLayer))
        , m_optimizerType(optimizerType)
        , m_argMax(nullptr)
    {
        MaxPool2D::loadFromFlash();
    }

    ~MaxPool2D() override
    {
        if (m_argMax != nullptr) {
            delete[] m_argMax;
            m_argMax = nullptr;
        }
    }

    auto forwardPass(const T* inputData, T* outputData, bool /* trainingflag */) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }
        if (!this->m_isLoaded) {
            return ErrorType::LayerNotInitialized;
        }

        const int C = m_header->channelsin;
        const int H_in = m_header->dimensioninput_y;
        const int W_in = m_header->dimensioninput_x;
        const int H_out = m_header->dimensionoutput_y;
        const int W_out = m_header->dimensionoutput_x;
        const int kH = m_header->kernelsize;
        const int kW = m_header->kernelsize;
        const int padH = m_header->padding;
        const int padW = m_header->padding;
        const int strideH = m_header->stride;
        const int strideW = m_header->stride;

        // allocate argmax storage
        size_t outSize = size_t(C) * H_out * W_out;
        delete[] m_argMax;
        m_argMax = new (std::nothrow) uint32_t[outSize];
        if (m_argMax == nullptr) {
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
                    m_argMax[outIdx] = maxIdx;
                }
            }
        }

        return ErrorType::OK;
    }

    auto backwardPass(const T* gradOutput, T* gradInput) -> ErrorType override
    {
        if (gradOutput == nullptr || gradInput == nullptr) {
            return ErrorType::UnknownError;
        }
        if (!this->m_isLoaded) {
            return ErrorType::LayerNotInitialized;
        }
        if (m_argMax == nullptr) {
            return ErrorType::UnknownError;
        }

        const int C = m_header->channelsin;
        const int H_in = m_header->dimensioninput_y;
        const int W_in = m_header->dimensioninput_x;
        const int H_out = m_header->dimensionoutput_y;
        const int W_out = m_header->dimensionoutput_x;

        // zero initialize grad_input
        size_t inSize = size_t(C) * H_in * W_in;
        for (size_t i = 0; i < inSize; ++i) {
            gradInput[i] = T(0);
        }

        // propagate gradients
        for (size_t outIdx = 0; outIdx < size_t(C) * H_out * W_out; ++outIdx) {
            uint32_t inIdx = m_argMax[outIdx];
            gradInput[inIdx] += gradOutput[outIdx];
        }

        return ErrorType::OK;
    }

    auto loadFromFlash() -> ErrorType override
    {
        this->m_isLoaded = true;
        return ErrorType::OK;
    }

    auto storeToFlash() -> ErrorType override
    {
        // not implemented for this version
        return ErrorType::OK;
    }

    auto getOutputSize() -> uint32_t override
    {
        return m_header->channelsin * m_header->dimensionoutput_x * m_header->dimensionoutput_y;
    }

    auto getInputSize() -> uint32_t override
    {
        return m_header->channelsin * m_header->dimensioninput_x * m_header->dimensioninput_y;
    }

private:
    void* m_ptrLayer;
    void* m_ptrData;
    Nmcm::MaxPool2d::NeuralNetwork_t* m_header;
    OptimizerID m_optimizerType;
    uint32_t* m_argMax;
};
} // namespace Edgeist

#endif // MAX_POOL_2D_H
