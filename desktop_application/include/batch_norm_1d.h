/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the 1D batch normalization layer
 */

#ifndef EDGEIST_BATCH_NORM_1D_H
#define EDGEIST_BATCH_NORM_1D_H

#include "layer.h"
#include "model_types.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {
template <typename T>
class Model;

/**
 * @brief 1D batch normalization layer
 *
 * Applies batch normalization to 1D input (typically features across a batch).
 * Normalizes the input to zero mean and unit variance, followed by a learnable
 * scale and shift.
 *
 * Helps stabilize and accelerate training by reducing internal covariate shift.
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class BatchNorm1D : public Layer<T> {
public:
    BatchNorm1D(Model<T>* model, void* headerPointer, void* dataPointer, const OptimizerID optimizerType)
        : Layer<T>(model)
        , m_ptrLayer(headerPointer)
        , m_ptrData(dataPointer)
        , m_header(static_cast<Nmcm::BatchNorm1d::NeuralNetwork*>(m_ptrLayer))
        , m_optimizerType(optimizerType)
    {
        BatchNorm1D::loadFromFlash();

        m_runningMean = new double[m_header->dimensionInputX];
        m_runningVar = new double[m_header->dimensionInputX];

        m_ptrFlashWeight = (static_cast<T*>(m_ptrData) + (m_header->weightsTrainableOffset / sizeof(T)));
        m_ptrFlashBias = (static_cast<T*>(m_ptrData) + (m_header->biasTrainableOffset / sizeof(T)));

        BatchNorm1D::loadFromFlash();
    }

    ~BatchNorm1D() override
    {
        delete[] m_weightPtr;
        m_weightPtr = nullptr;
        delete[] m_biasPtr;
        m_biasPtr = nullptr;

        delete m_runningMean;
        m_runningMean = nullptr;
        delete m_runningVar;
        m_runningVar = nullptr;
    }

    auto forwardPass(const T* inputData, T* outputData, bool trainingFlag) -> ErrorType override
    {
        // Use ether data from flash or SRAM based on trainingflag
        T* ptrWeight = nullptr;
        T* ptrBias = nullptr;
        if (trainingFlag) {
            // check if layer is loaded
            if (!this->m_isLoaded) {
                return ErrorType::LayerNotInitialized;
            }
            ptrWeight = m_weightPtr->m_data;
            ptrBias = m_biasPtr->m_data;
        } else {
            ptrWeight = m_ptrFlashWeight;
            ptrBias = m_ptrFlashBias;
            m_initRunningStats = false;
        }

        const int nrOfInputs = m_header->dimensionInputX;

        float eps = 1e-5f;

        double mean = double(0.0);
        double var = double(1.0);

        // calculate average
        for (int i = 0; i < nrOfInputs; i++) {
            mean += inputData[i];
        }
        mean = mean / nrOfInputs;

        // calculate variance
        for (int i = 0; i < nrOfInputs; i++) {
            T diff = inputData[i] - mean;
            var += diff * diff;
        }
        var = var / nrOfInputs;

        if (m_initRunningStats == false) {
            // initialize with the values from buffer
            for (int i = 0; i < nrOfInputs; i++) {
                m_runningMean[i] = ptrWeight[i + nrOfInputs];
                m_runningVar[i] = ptrBias[i + nrOfInputs];
            }
            m_initRunningStats = true;
        }

        if (trainingFlag) {
            // forward pass with training
            float momentum = 0.1f;
            for (int i = 0; i < nrOfInputs; i++) {
                m_runningMean[i] = (1 - momentum) * m_runningMean[i] + momentum * mean;
                m_runningVar[i] = (1 - momentum) * m_runningVar[i] + momentum * var;
            }

            m_normalizedInput = new T[nrOfInputs];
            m_input = new T[nrOfInputs];

            // normalize + scale + move
            for (int i = 0; i < nrOfInputs; i++) {
                double norm = (double(inputData[i]) - m_runningMean[i]) / std::sqrt(m_runningVar[i] + eps);

                // save input and normaliced input
                m_normalizedInput[i] = norm;
                m_input[i] = inputData[i];

                T gamma = ptrWeight[i];
                T beta = ptrBias[i];
                outputData[i] = gamma * norm + beta;
            }
            // store values for backwardpass

        } else {
            // normalize + scale + move
            for (int i = 0; i < nrOfInputs; i++) {
                T gamma = ptrWeight[i];
                T beta = ptrBias[i];
                double norm = (double(inputData[i]) - m_runningMean[i]) / std::sqrt(m_runningVar[i] + eps);
                outputData[i] = gamma * norm + beta;
            }
        }

        return ErrorType::OK;
    }

    auto backwardPass(const T* inputData, T* outputData) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::InvalidPointer;
        }

        if (m_input == nullptr || m_normalizedInput == nullptr) {
            return ErrorType::MissingCachedInputs;
        }

        int nrOfInputs = m_header->dimensionInputX;
        float eps = 1e-5f;

        // cache
        T* dL_dgamma = new T[nrOfInputs];
        T* dL_dbeta = new T[nrOfInputs];
        T* dL_dnorm = new T[nrOfInputs]; // dL/dx

        // init
        for (int i = 0; i < nrOfInputs; ++i) {
            dL_dgamma[i] = inputData[i] * m_normalizedInput[i]; // dL/dy = dL/dy * x
            dL_dbeta[i] = inputData[i]; // dL/db = dL/dy
            dL_dnorm[i] = inputData[i] * m_weightPtr->getData(i); // dL/dx = dL/dy * y
        }

        // calculate helper values
        double sum_dnorm = 0.0;
        double sum_dnorm_norm = 0.0;
        for (int i = 0; i < nrOfInputs; ++i) {
            sum_dnorm += dL_dnorm[i];
            sum_dnorm_norm += dL_dnorm[i] * m_normalizedInput[i];
        }

        // calculate dL/dx
        for (int i = 0; i < nrOfInputs; ++i) {
            double std_inv = 1.0 / std::sqrt(m_runningVar[i] + eps);
            double term1 = nrOfInputs * dL_dnorm[i];
            double term2 = sum_dnorm;
            double term3 = m_normalizedInput[i] * sum_dnorm_norm;

            outputData[i] = (1.0 / nrOfInputs) * std_inv * (term1 - term2 - term3);
        }

        // accumulate within gradient buffer
        for (int i = 0; i < nrOfInputs; ++i) {
            m_ptrWeightGradient[i] += dL_dgamma[i]; // for gamma
            m_ptrBiasGradient[i] += dL_dbeta[i]; // for beta
        }

        // cleanup
        delete[] dL_dgamma;
        dL_dgamma = nullptr;
        delete[] dL_dbeta;
        dL_dbeta = nullptr;
        delete[] dL_dnorm;
        dL_dnorm = nullptr;
        // output_data is output

        delete[] m_input;
        m_input = nullptr;
        delete[] m_normalizedInput;
        m_normalizedInput = nullptr;

        return ErrorType::OK;
    }

    // initializes mean and variance before a mini batch
    auto initGradients() -> ErrorType override
    {
        if (m_ptrWeightGradient != nullptr || m_ptrBiasGradient != nullptr) {
            return ErrorType::UnknownError;
        }

        // create arrays dynamically and initialize them with 0.0
        const uint32_t sizeWeights = m_header->dimensionInputX;
        const uint32_t sizeBias = m_header->dimensionInputX;

        m_ptrWeightGradient = new T[sizeWeights];
        m_ptrBiasGradient = new T[sizeBias];

        for (uint32_t i = 0; i < sizeWeights; ++i) {
            m_ptrWeightGradient[i] = T(1.0);
        }

        for (uint32_t i = 0; i < sizeBias; ++i) {
            m_ptrBiasGradient[i] = T(0.0);
        }
        return ErrorType::OK;
    }

    // delete mean and variance after minibatch
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

    auto update(uint32_t batchsize) -> ErrorType override
    {
        if (m_ptrWeightGradient == nullptr || m_ptrBiasGradient == nullptr) {
            return ErrorType::UnknownError;
        }

        // SGD-Update of W and b
        for (Nmcm::BatchNorm1d::DimensionInputX i = 0; i < m_header->dimensionInputX; ++i) {
            m_weightPtr->update(i, m_ptrWeightGradient[i] / batchsize, this->m_model->m_learningRate, m_timestep);
            m_biasPtr->update(i, m_ptrBiasGradient[i] / batchsize, this->m_model->m_learningRate, m_timestep);
        }
        m_timestep++;

        return ErrorType::OK;
    }

    auto loadFromFlash() -> ErrorType override
    {
        switch (m_optimizerType) {
        case OptimizerID::SGD:
            // init weights
            m_weightPtr = new OptimizerSGD<T>;
            m_weightPtr->init(m_header->weightsAmountTrainable);
            // init bias
            m_biasPtr = new OptimizerSGD<T>;
            m_biasPtr->init(m_header->biasAmountTrainable);
            break;

        case OptimizerID::Momentum:
            // init weights
            m_weightPtr = new OptimizerMomentum<T>;
            m_weightPtr->init(m_header->weightsAmountTrainable);
            // init bias
            m_biasPtr = new OptimizerMomentum<T>;
            m_biasPtr->init(m_header->biasAmountTrainable);
            break;

        case OptimizerID::ADAM:
            // init weights
            m_weightPtr = new OptimizerAdam<T>;
            m_weightPtr->init(m_header->weightsAmountTrainable);
            // init bias
            m_biasPtr = new OptimizerAdam<T>;
            m_biasPtr->init(m_header->biasAmountTrainable);
            break;
        }

        // init weights and biases
        for (size_t i = 0; i < m_header->weightsAmountTrainable; i++) {
            m_weightPtr->setData(i, *(static_cast<T*>(m_ptrData) + (m_header->weightsTrainableOffset / sizeof(T)) + i));
        }
        for (size_t i = 0; i < m_header->biasAmountTrainable; i++) {
            m_biasPtr->setData(i, *(static_cast<T*>(m_ptrData) + (m_header->biasTrainableOffset / sizeof(T)) + i));
        }

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
        return m_header->dimensionOutputX;
    }

    auto getInputSize() -> uint32_t override
    {
        return m_header->dimensionInputX;
    }

private:
    void* m_ptrLayer;
    void* m_ptrData;
    Nmcm::BatchNorm1d::NeuralNetwork* m_header;
    OptimizerID m_optimizerType;

    double* m_runningMean = nullptr;
    double* m_runningVar = nullptr;

    // TODO fix datatypes
    T* m_normalizedInput = nullptr;
    T* m_input = nullptr;

    // flag for first run, to init running mean/var
    bool m_initRunningStats = false;

    T* m_ptrFlashWeight = nullptr;
    T* m_ptrFlashBias = nullptr;

    T* m_ptrWeightGradient = nullptr;
    T* m_ptrBiasGradient = nullptr;

    OptimizerBase<T>* m_weightPtr;
    OptimizerBase<T>* m_biasPtr;

    uint32_t m_timestep = 1;
};
} // namespace Edgeist

#endif // EDGEIST_BATCH_NORM_1D_H
