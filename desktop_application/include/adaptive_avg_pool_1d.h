/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the 1D adaptive average pooling layer
 */

#ifndef EDGEIST_ADAPTIVE_AVG_POOL_1D_H
#define EDGEIST_ADAPTIVE_AVG_POOL_1D_H

#include <cmath>

#include "layer.h"
#include "model_structs.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {
/**
 * @brief 1D adaptive average pooling layer
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
class AdaptiveAvgPool1D : public Layer<T> {
public:
    AdaptiveAvgPool1D(Model<T>* model, void* headerPointer, void* dataPointer, const OptimizerID optimizerType)
        : Layer<T>(model)
        , m_ptrLayer(headerPointer)
        , m_ptrData(dataPointer)
        , m_header(static_cast<Nmcm::AdaptiveAvgPool1d::NeuralNetwork*>(m_ptrLayer))
        , m_optimizerType(optimizerType)
    {
        AdaptiveAvgPool1D::loadFromFlash();
    }

    auto forwardPass(const T* inputData, T* outputData, bool /* trainingflag */) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }
        if (!this->m_isLoaded) {
            return ErrorType::LayerNotInitialized;
        }

        const int C = m_header->channelsIn;
        const int W_in = m_header->dimensionInputX;
        const int W_out = m_header->dimensionOutputX;

        for (int c = 0; c < C; ++c) {
            for (int ox = 0; ox < W_out; ++ox) {
                int x_start = std::floor(ox * W_in / static_cast<float>(W_out));
                int x_end = std::ceil((ox + 1) * W_in / static_cast<float>(W_out));

                T sum = 0;
                int count = 0;
                for (int ix = x_start; ix < x_end; ++ix) {
                    size_t in_idx = size_t(c) * W_in + ix;
                    sum += inputData[in_idx];
                    ++count;
                }

                size_t out_idx = size_t(c) * W_out + ox;
                outputData[out_idx] = count > 0 ? sum / static_cast<T>(count) : T(0);
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

        const int C = m_header->channelsIn;
        const int W_in = m_header->dimensionInputX;
        const int W_out = m_header->dimensionOutputX;

        size_t inSize = size_t(C) * W_in;
        for (size_t i = 0; i < inSize; ++i) {
            gradInput[i] = T(0);
        }

        for (int c = 0; c < C; ++c) {
            for (int ox = 0; ox < W_out; ++ox) {
                int x_start = std::floor(ox * W_in / static_cast<float>(W_out));
                int x_end = std::ceil((ox + 1) * W_in / static_cast<float>(W_out));

                size_t out_idx = size_t(c) * W_out + ox;
                T grad = gradOutput[out_idx];
                int count = x_end - x_start;
                T grad_val = count > 0 ? grad / static_cast<T>(count) : T(0);

                for (int ix = x_start; ix < x_end; ++ix) {
                    size_t in_idx = size_t(c) * W_in + ix;
                    gradInput[in_idx] += grad_val;
                }
            }
        }

        return ErrorType::OK;
    }

    // init dropout mask for mini batch
    auto initGradients() -> ErrorType override
    {
        // TODO
        return ErrorType::OK;
    }

    // delete dropout mask
    auto deleteGradients() -> ErrorType override
    {
        // TODO
        return ErrorType::OK;
    }

    auto loadFromFlash() -> ErrorType override
    {
        this->m_isLoaded = true;
        return ErrorType::OK;
    }

    auto storeToFlash() -> ErrorType override
    {
        // not implemented
        return ErrorType::OK;
    }

    auto getOutputSize() -> uint32_t override { return m_header->channelsIn * m_header->dimensionOutputX; }

    auto getInputSize() -> uint32_t override { return m_header->channelsIn * m_header->dimensionInputX; }

private:
    void* m_ptrLayer;
    void* m_ptrData;
    Nmcm::AdaptiveAvgPool1d::NeuralNetwork* m_header;
    OptimizerID m_optimizerType;
};
} // namespace Edgeist

#endif // EDGEIST_ADAPTIVE_AVG_POOL_1D_H
