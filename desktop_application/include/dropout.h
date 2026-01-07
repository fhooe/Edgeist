/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the dropout layer
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
 * @brief Dropout layer
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
        , m_ptrLayer(headerPointer)
        , m_ptrData(dataPointer)
        , m_header(static_cast<Nmcm::Dropout::NeuralNetwork_t*>(m_ptrLayer))
        , m_optimizerType(optimizerType)
        , m_rng(std::random_device {}())
    {
        Dropout::loadFromFlash();
    }

    auto forwardPass(const T* inputData, T* outputData, const bool trainingFlag) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }
        if (!this->m_isLoaded) {
            return ErrorType::LayerNotInitialized;
        }

        const size_t size = m_header->dimensioninput_x;

        if (trainingFlag) {
            if (m_dropoutMask.size() != size) {
                return ErrorType::DropoutMaskMissing;
            }

            for (size_t i = 0; i < size; ++i) {
                outputData[i] = inputData[i] * m_dropoutMask[i];
            }
        } else {
            for (size_t i = 0; i < size; ++i) {
                outputData[i] = inputData[i]; // do not change in inference mode
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

        const size_t size = m_header->dimensioninput_x;

        if (m_dropoutMask.size() != size) {
            return ErrorType::DropoutMaskMissing;
        }

        for (size_t i = 0; i < size; ++i) {
            gradInput[i] = gradOutput[i] * m_dropoutMask[i];
        }

        return ErrorType::OK;
    }

    auto initGradients() -> ErrorType override
    {
        const auto size = static_cast<size_t>(m_header->dimensioninput_x);
        const float rate = m_header->dropoutrate;

        std::uniform_real_distribution<float> dist(0.0F, 1.0F);
        m_dropoutMask.resize(size);

        for (size_t i = 0; i < size; ++i) {
            bool keep = dist(m_rng) >= rate;
            m_dropoutMask[i] = keep ? T(1.0F) / (1.0F - rate) : T(0);
        }

        return ErrorType::OK;
    }

    auto deleteGradients() -> ErrorType override
    {
        m_dropoutMask.clear();
        return ErrorType::OK;
    }

    auto loadFromFlash() -> ErrorType override
    {
        this->m_isLoaded = true;
        return ErrorType::OK;
    }

    auto storeToFlash() -> ErrorType override
    {
        // Not implemented
        return ErrorType::OK;
    }

    auto getOutputSize() -> uint32_t override
    {
        return m_header->dimensionoutput_x;
    }

    auto getInputSize() -> uint32_t override
    {
        return m_header->dimensioninput_x;
    }

private:
    void* m_ptrLayer;
    void* m_ptrData;
    Nmcm::Dropout::NeuralNetwork_t* m_header;
    OptimizerID m_optimizerType;

    std::vector<T> m_dropoutMask;
    std::mt19937 m_rng;
};
} // namespace Edgeist

#endif // DROPOUT_H
