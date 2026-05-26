/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the fully connected (dense) linear layer
 */

#ifndef EDGEIST_LINEAR_H
#define EDGEIST_LINEAR_H

#include "model_structs.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"
#include <memory.h>
namespace Edgeist {
/**
 * @brief Fully connected (dense) linear layer
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
        , m_ptrLayer(static_cast<T*>(headerPointer))
        , m_ptrData(static_cast<T*> (dataPointer))
        , m_header(static_cast<Nmcm::Linear::NeuralNetwork*>(headerPointer))
        , m_optimizerType(optimizerType)
    {
        m_ptrWeightFrozen = (m_ptrLayer + (m_header->weightsFrozenOffset * sizeof(T)));
        m_ptrBiasFrozen = (m_ptrLayer + (m_header->biasFrozenOffset * sizeof(T)));

        m_ptrFlashWeight = (m_ptrData + (m_header->weightsTrainableOffset / sizeof(T)));
        m_ptrFlashBias = (m_ptrData + (m_header->biasTrainableOffset / sizeof(T)));

        Linear::loadFromFlash();
    }

    // DTor: frees dynamically allocated memory
    ~Linear() override
    {
        // free memory
        if (m_ptrWeight != nullptr) {
            m_ptrWeight.reset();
        }

        if (m_ptrBias != nullptr) {
            m_ptrBias.reset();
        }

        if (this->m_inputData != nullptr) {
            delete[] this->m_inputData;
            this->m_inputData = nullptr;
        }

        if (m_ptrWeightGradient != nullptr) {
            m_ptrWeightGradient.reset();
        }

        if (m_ptrBiasGradient != nullptr) {
            m_ptrBiasGradient.reset();
        }
    }

    // executes forward pass and writes the result to the output
    ErrorType forwardPass(const T* inputData, T* outputData, bool trainingFlag) override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }

        // use ether data from flash or SRAM based on trainingflag
        const T* ptrWeight = nullptr;
        const T* ptrBias = nullptr;
        if (trainingFlag) {
            // check if layer is loaded
            if (!this->m_isLoaded) {
                return ErrorType::LayerNotInitialized;
            }
            ptrWeight = m_ptrWeight->m_data.data();
            ptrBias = m_ptrBias->m_data.data();
        } else {
            ptrWeight = m_ptrFlashWeight;
            ptrBias = m_ptrFlashBias;
        }

        if (trainingFlag) {
            // get memory for training
            if (this->m_inputData == nullptr) {
                this->m_inputData = new T[m_header->dimensionInputX];
            }
            for (uint32_t i = 0; i < m_header->dimensionInputX; i++) {
                this->m_inputData[i] = inputData[i];
            }
        }

        // calc result
        // step through outputs
        for (uint32_t b = 0; b < m_header->dimensionOutputX; b++) {
            auto temp = T(0);
            for (uint32_t i = 0; i < m_header->dimensionInputX; i++) {
                temp += inputData[i] * ptrWeight[(b * (m_header->dimensionInputX)) + i];
            }

            temp = ptrBias[b] + temp;
            outputData[b] = temp;
        }

        return ErrorType::OK;
    }

    // executes the backward pass and calculates the gradient for the layer before
    ErrorType backwardPass(const T* inputData, T* outputData) override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }

        if (this->m_inputData == nullptr) {
            return ErrorType::UnknownError;
        }

        // do backward pass
        T* dL_dW = new T[m_header->dimensionOutputX * m_header->dimensionInputX];

        for (size_t i = 0; i < m_header->dimensionOutputX; ++i) {
            for (size_t j = 0; j < m_header->dimensionInputX; ++j) {
                dL_dW[i * m_header->dimensionInputX + j] = inputData[i] * this->m_inputData[j];
            }
        }

        // 2. gradient for bias
        T* dL_db = new T[m_header->dimensionOutputX];

        for (size_t i = 0; i < m_header->dimensionOutputX; ++i) {
            dL_db[i] = inputData[i];
        }

        // 3. gradient for input x (backprop through layer)
        for (size_t j = 0; j < m_header->dimensionInputX; ++j) {
            outputData[j] = 0.0f;
            for (size_t i = 0; i < m_header->dimensionOutputX; ++i) {
                outputData[j] += m_ptrWeight->getData((i * (m_header->dimensionInputX)) + j) * inputData[i];
            }
        }

        // update gradients
        for (size_t i = 0; i < m_header->dimensionOutputX; ++i) {
            for (size_t j = 0; j < m_header->dimensionInputX; ++j) {
                m_ptrWeightGradient[i * m_header->dimensionInputX + j] += dL_dW[i * m_header->dimensionInputX + j];
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
    ErrorType initGradients() override
    {
        if (m_ptrWeightGradient != nullptr || m_ptrBiasGradient != nullptr) {
            return ErrorType::UnknownError;
        }

        // create arrays dynamically and initialize them with 0.0
        uint32_t sizeWeights = m_header->dimensionOutputX * m_header->dimensionInputX;
        uint32_t sizeBias = m_header->dimensionOutputX;

        m_ptrWeightGradient = std::make_unique<T[]>(sizeWeights);
        m_ptrBiasGradient = std::make_unique<T[]>(sizeBias);

        for (uint32_t i = 0; i < sizeWeights; ++i) {
            m_ptrWeightGradient[i] = T(0.0);
        }

        for (uint32_t i = 0; i < sizeBias; ++i) {
            m_ptrBiasGradient[i] = T(0.0);
        }

        return ErrorType::OK;
    }

    // delete gradient and free memory
    ErrorType deleteGradients() override
    {

        if (m_ptrWeightGradient != nullptr) {
            m_ptrWeightGradient.reset();
        }

        if (m_ptrBiasGradient != nullptr) {
            m_ptrBiasGradient.reset();
        }


        return ErrorType::OK;
    }

    // update weights and biases
    ErrorType update(uint32_t batchsize) override
    {
        if (m_ptrWeightGradient == nullptr || m_ptrBiasGradient == nullptr) {
            return ErrorType::UnknownError;
        }

        // SGD-Update of W and b
        for (size_t i = 0; i < m_header->dimensionOutputX; ++i) {
            for (size_t j = 0; j < m_header->dimensionInputX; ++j) {
                m_ptrWeight->update((i * (m_header->dimensionInputX)) + j, m_ptrWeightGradient[i * m_header->dimensionInputX + j] / batchsize, this->m_model->m_learningRate, m_timestep);
            }
            m_ptrBias->update(i, m_ptrBiasGradient[i] / batchsize, this->m_model->m_learningRate, m_timestep);
        }
        m_timestep++;
        return ErrorType::OK;
    }

    // Loads the trainable values from Flash into SRAM
    ErrorType loadFromFlash() override
    {
        switch (m_optimizerType) {
        case OptimizerID::SGD:
            m_ptrWeight = std::make_unique<OptimizerSGD<T>>();
            m_ptrWeight->init(m_header->weightsAmountTrainable);
            m_ptrBias   = std::make_unique<OptimizerSGD<T>>();
            m_ptrBias->init(m_header->biasAmountTrainable);           
            break;

        case OptimizerID::Momentum:
            m_ptrWeight = std::make_unique<OptimizerMomentum<T>>();
            m_ptrWeight->init(m_header->weightsAmountTrainable);
            m_ptrBias   = std::make_unique<OptimizerMomentum<T>>();
            m_ptrBias->init(m_header->biasAmountTrainable);            
            break;

        case OptimizerID::ADAM:
            m_ptrWeight = std::make_unique<OptimizerAdam<T>>();
            m_ptrWeight->init(m_header->weightsAmountTrainable);
            m_ptrBias   = std::make_unique<OptimizerAdam<T>>();
            m_ptrBias->init(m_header->biasAmountTrainable);
            break;
        }

        // init weights and biases
        for (size_t i = 0; i < m_header->weightsAmountTrainable; i++) {
            m_ptrWeight->setData(i, *m_ptrData + (m_header->weightsTrainableOffset / sizeof(T)) + i);
        }
        for (size_t i = 0; i < m_header->biasAmountTrainable; i++) {
            m_ptrBias->setData(i, *m_ptrData + (m_header->biasTrainableOffset / sizeof(T)) + i);
        }

        this->m_isLoaded = true;
        return ErrorType::OK;
    }

    // Stores the trainable values from SRAM to Flash
    ErrorType storeToFlash() override
    {
        this->m_isLoaded = false;
        // Not implemented in this version, will only become relevant on the uController
        return ErrorType::UnknownError;
    }

    uint32_t getOutputSize() override
    {
        return uint32_t(m_header->dimensionOutputX);
    }

    uint32_t getInputSize() override
    {
        return uint32_t(m_header->dimensionInputX);
    }

    [[nodiscard]] auto header() const -> const Nmcm::Linear::NeuralNetwork&
    {
        return *m_header;
    }

private:
    //void* m_ptrLayer;
    T* m_ptrLayer;
    T* m_ptrData;

    Nmcm::Linear::NeuralNetwork* m_header;

    // chosen optimizer
    OptimizerID m_optimizerType;

    T* m_ptrWeightFrozen = nullptr;
    T* m_ptrBiasFrozen = nullptr;

    T const *  m_ptrFlashWeight = nullptr;
    T const *  m_ptrFlashBias = nullptr;

    //T* m_ptrWeightGradient = nullptr;
    //T* m_ptrBiasGradient = nullptr;
    std::unique_ptr<T[]> m_ptrWeightGradient;
    std::unique_ptr<T[]> m_ptrBiasGradient;

    // vector with the trainable weights in SRAM
    std::unique_ptr<OptimizerBase<T>> m_ptrWeight;
    std::unique_ptr<OptimizerBase<T>> m_ptrBias;
    //OptimizerBase<T>* m_ptrWeight;
    //OptimizerBase<T>* m_ptrBias;

    uint32_t m_timestep = 1;
};
} // namespace Edgeist

#endif // EDGEIST_LINEAR_H
