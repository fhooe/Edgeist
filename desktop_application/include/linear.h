/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the fully connected (dense) linear layer.
 */

#ifndef LINEAR_H
#define LINEAR_H

#include "Modelstructs.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {
/**
 * @brief Fully connected (dense) linear layer.
 *
 * Applies a linear transformation: `output = input * weight^T + bias`.
 * Commonly used for classification or regression tasks.
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class Linear : public Layer<T> {
public:
    // CTor: Init the ReLU-Layer with pointers to config data and the chosen optimizer
    Linear(Model<T>* model, void* headerPointer, void* dataPointer, const OptimizerID optimizerType)
        : Layer<T>(model)
        , m_ptrLayer(headerPointer)
        , m_ptrData(dataPointer)
        , m_header(static_cast<Neural_Network_Linear_t*>(m_ptrLayer))
        , m_optimizerType(optimizerType)
    {
        m_ptrWeightFrozen = static_cast<T*>(m_ptrLayer) + m_header->weights_frozen_offset * sizeof(T);
        m_ptrBiasFrozen = static_cast<T*>(m_ptrLayer) + m_header->bias_frozen_offset * sizeof(T);

        m_ptrFlashWeight = (static_cast<T*>(m_ptrData) + (m_header->weights_trainable_offset / sizeof(T)));
        m_ptrFlashBias = (static_cast<T*>(m_ptrData) + (m_header->bias_trainable_offset / sizeof(T)));

        Linear::loadFromFlash();
    }

    // DTor: frees dynamically allocated memory
    ~Linear() override
    {
        // free memory
        if (m_ptrWeight != nullptr) {
            delete[] m_ptrWeight;
            m_ptrWeight = nullptr;
        }

        if (m_ptrBias != nullptr) {
            delete[] m_ptrBias;
            m_ptrBias = nullptr;
        }

        if (this->m_inputData != nullptr) {
            delete[] this->m_inputData;
            this->m_inputData = nullptr;
        }

        if (m_ptrWeightGradient != nullptr) {
            delete[] m_ptrWeightGradient;
            m_ptrWeightGradient = nullptr;
        }

        if (m_ptrBiasGradient != nullptr) {
            delete[] m_ptrBiasGradient;
            m_ptrBiasGradient = nullptr;
        }
    }

    // executes forward pass and writes the result to the output
    auto forwardPass(const T* inputData, T* outputData, bool trainingFlag) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }

        // use ether data from flash or SRAM based on trainingflag
        T* ptrWeight = nullptr;
        T* ptrBias = nullptr;
        if (trainingFlag) {
            // check if layer is loaded
            if (!this->m_isLoaded) {
                return ErrorType::LayerNotInitialized;
            }
            ptrWeight = m_ptrWeight->m_data;
            ptrBias = m_ptrBias->m_data;
        } else {
            ptrWeight = m_ptrFlashWeight;
            ptrBias = m_ptrFlashBias;
        }

        if (trainingFlag) {
            // get memory for training
            if (this->m_inputData == nullptr) {
                this->m_inputData = new T[m_header->dimensioninput_x];
            }
            for (uint32_t i = 0; i < m_header->dimensioninput_x; i++) {
                this->m_inputData[i] = inputData[i];
            }
        }

        // calc result
        // step through outputs
        for (uint32_t b = 0; b < m_header->dimensionoutput_x; b++) {
            auto temp = T(0);
            for (uint32_t i = 0; i < m_header->dimensioninput_x; i++) {
                temp += inputData[i] * ptrWeight[(b * (m_header->dimensioninput_x)) + i];
            }

            temp = ptrBias[b] + temp;
            outputData[b] = temp;
        }

        return ErrorType::OK;
    }

    // executes the backward pass and calculates the gradient for the layer before
    auto backwardPass(const T* inputData, T* outputData) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }

        if (this->m_inputData == nullptr) {
            return ErrorType::UnknownError;
        }

        // do backward pass
        T* dL_dW = new T[m_header->dimensionoutput_x * m_header->dimensioninput_x];
        for (size_t i = 0; i < m_header->dimensionoutput_x; ++i) {
            for (size_t j = 0; j < m_header->dimensioninput_x; ++j) {
                dL_dW[i * m_header->dimensioninput_x + j] = inputData[i] * this->m_inputData[j];
            }
        }

        // 2. gradient for bias
        T* dL_db = new T[m_header->dimensionoutput_x];
        for (size_t i = 0; i < m_header->dimensionoutput_x; ++i) {
            dL_db[i] = inputData[i];
        }

        // 3. gradient for input x (backprop through layer)
        for (size_t j = 0; j < m_header->dimensioninput_x; ++j) {
            outputData[j] = 0.0f;
            for (size_t i = 0; i < m_header->dimensionoutput_x; ++i) {
                outputData[j] += m_ptrWeight->getData((i * (m_header->dimensioninput_x)) + j) * inputData[i];
            }
        }

        // update gradients
        for (size_t i = 0; i < m_header->dimensionoutput_x; ++i) {
            for (size_t j = 0; j < m_header->dimensioninput_x; ++j) {
                m_ptrWeightGradient[i * m_header->dimensioninput_x + j] += dL_dW[i * m_header->dimensioninput_x + j];
            }
            m_ptrBiasGradient[i] += dL_db[i];
        }

        if (this->m_inputData != nullptr) {
            delete[] this->m_inputData;
            this->m_inputData = nullptr;
        }

        delete[] dL_dW;
        dL_dW = nullptr;
        delete[] dL_db;
        dL_db = nullptr;

        return ErrorType::OK;
    }

    // init gradient, reserve and init memory
    auto initGradients() -> ErrorType override
    {
        if (m_ptrWeightGradient != nullptr || m_ptrBiasGradient != nullptr) {
            return ErrorType::UnknownError;
        }

        // create arrays dynamically and initialize them with 0.0
        uint32_t sizeWeights = m_header->dimensionoutput_x * m_header->dimensioninput_x;
        uint32_t sizeBias = m_header->dimensionoutput_x;

        m_ptrWeightGradient = new T[sizeWeights];
        m_ptrBiasGradient = new T[sizeBias];

        for (uint32_t i = 0; i < sizeWeights; ++i) {
            m_ptrWeightGradient[i] = T(0.0);
        }

        for (uint32_t i = 0; i < sizeBias; ++i) {
            m_ptrBiasGradient[i] = T(0.0);
        }

        return ErrorType::OK;
    }

    // delete gradient and free memory
    auto deleteGradients() -> ErrorType override
    {
        if (m_ptrWeightGradient != nullptr) {
            delete[] m_ptrWeightGradient;
            m_ptrWeightGradient = nullptr;
        }

        if (m_ptrBiasGradient != nullptr) {
            delete[] m_ptrBiasGradient;
            m_ptrBiasGradient = nullptr;
        }

        return ErrorType::OK;
    }

    // update weights and biases
    auto update(uint32_t batchsize) -> ErrorType override
    {
        if (m_ptrWeightGradient == nullptr || m_ptrBiasGradient == nullptr) {
            return ErrorType::UnknownError;
        }

        // SGD-Update of W and b
        for (size_t i = 0; i < m_header->dimensionoutput_x; ++i) {
            for (size_t j = 0; j < m_header->dimensioninput_x; ++j) {
                m_ptrWeight->update((i * (m_header->dimensioninput_x)) + j, m_ptrWeightGradient[i * m_header->dimensioninput_x + j] / batchsize, this->m_model->m_learningRate, m_timestep);
            }
            m_ptrBias->update(i, m_ptrBiasGradient[i] / batchsize, this->m_model->m_learningRate, m_timestep);
        }
        m_timestep++;
        return ErrorType::OK;
    }

    // Loads the trainable values from Flash into SRAM
    auto loadFromFlash() -> ErrorType override
    {
        switch (m_optimizerType) {
        case OptimizerID::SGD:
            // init weights
            m_ptrWeight = new OptimizerSGD<T>;
            m_ptrWeight->init(m_header->weights_amount_trainable);
            // init bias
            m_ptrBias = new OptimizerSGD<T>;
            m_ptrBias->init(m_header->bias_amount_trainable);
            break;

        case OptimizerID::Momentum:
            // init weights
            m_ptrWeight = new OptimizerMomentum<T>;
            m_ptrWeight->init(m_header->weights_amount_trainable);
            // init bias
            m_ptrBias = new OptimizerMomentum<T>;
            m_ptrBias->init(m_header->bias_amount_trainable);
            break;

        case OptimizerID::ADAM:
            // init weights
            m_ptrWeight = new OptimizerAdam<T>;
            m_ptrWeight->init(m_header->weights_amount_trainable);
            // init bias
            m_ptrBias = new OptimizerAdam<T>;
            m_ptrBias->init(m_header->bias_amount_trainable);
            break;
        }

        // init weights and biases
        for (size_t i = 0; i < m_header->weights_amount_trainable; i++) {
            m_ptrWeight->setData(i, *(static_cast<T*>(m_ptrData) + (m_header->weights_trainable_offset / sizeof(T)) + i));
        }
        for (size_t i = 0; i < m_header->bias_amount_trainable; i++) {
            m_ptrBias->setData(i, *(static_cast<T*>(m_ptrData) + (m_header->bias_trainable_offset / sizeof(T)) + i));
        }

        this->m_isLoaded = true;
        return ErrorType::OK;
    }

    // Stores the trainable values from SRAM to Flash
    auto storeToFlash() -> ErrorType override
    {
        this->m_isLoaded = false;
        // Not implemented in this version, will only become relevant on the uController
        return ErrorType::UnknownError;
    }

    auto getOutputSize() -> uint32_t override
    {
        return uint32_t(m_header->dimensionoutput_x);
    }

    auto getInputSize() -> uint32_t override
    {
        return uint32_t(m_header->dimensioninput_x);
    }

    [[nodiscard]] auto header() const -> const Neural_Network_Linear_t&
    {
        return *m_header;
    }

private:
    void* m_ptrLayer;
    void* m_ptrData;

    Neural_Network_Linear_t* m_header;

    // chosen optimizer
    OptimizerID m_optimizerType;

    T* m_ptrWeightFrozen = nullptr;
    T* m_ptrBiasFrozen = nullptr;

    T* m_ptrFlashWeight = nullptr;
    T* m_ptrFlashBias = nullptr;

    T* m_ptrWeightGradient = nullptr;
    T* m_ptrBiasGradient = nullptr;

    // vector with the trainable weights in SRAM
    OptimizerBase<T>* m_ptrWeight;
    OptimizerBase<T>* m_ptrBias;

    uint32_t m_timestep = 1;
};
} // namespace Edgeist

#endif // LINEAR_H
