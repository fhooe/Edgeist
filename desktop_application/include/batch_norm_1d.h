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
#include <cstdint>
#include <memory.h>
#include <memory>

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
        , m_ptrData(static_cast<T*> (dataPointer))
        , m_header(static_cast<Nmcm::BatchNorm1d::NeuralNetwork*>(headerPointer))
        , m_optimizerType(optimizerType)
    {
        BatchNorm1D::loadFromFlash();

        //m_runningMean = new double[m_header->dimensionInputX];
        //m_runningVar = new double[m_header->dimensionInputX];

        m_runningMean =  std::make_unique<double[]>(m_header->dimensionInputX);
        m_runningVar =  std::make_unique<double[]>(m_header->dimensionInputX);

        m_ptrFlashWeight = (m_ptrData + (m_header->weightsTrainableOffset / sizeof(T)));
        m_ptrFlashBias = (m_ptrData + (m_header->biasTrainableOffset / sizeof(T)));

        BatchNorm1D::loadFromFlash();
    }

    ErrorType forwardPass(const T* inputData, T* outputData, bool trainingFlag) override
    {
        // Use ether data from flash or SRAM based on trainingflag
        const T* ptrWeight = nullptr;
        const T* ptrBias = nullptr;
        if (trainingFlag) {
            // check if layer is loaded
            if (!this->m_isLoaded) {
                return ErrorType::LayerNotInitialized;
            }
            ptrWeight = m_weightPtr->m_data.data();
            ptrBias = m_biasPtr->m_data.data();
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

    ErrorType backwardPass(const T* inputData, T* outputData) override
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
    ErrorType initGradients() override
    {
        if (m_ptrWeightGradient != nullptr || m_ptrBiasGradient != nullptr) {
            return ErrorType::UnknownError;
        }

        // create arrays dynamically and initialize them with 0.0
        const uint32_t sizeWeights = m_header->dimensionInputX;
        const uint32_t sizeBias = m_header->dimensionInputX;

        m_ptrWeightGradient = std::make_unique<T[]>(sizeWeights);
        m_ptrBiasGradient = std::make_unique<T[]>(sizeBias);

        for (uint32_t i = 0; i < sizeWeights; ++i) {
            m_ptrWeightGradient[i] = T(1.0);
        }

        for (uint32_t i = 0; i < sizeBias; ++i) {
            m_ptrBiasGradient[i] = T(0.0);
        }
        return ErrorType::OK;
    }

    ErrorType update(uint32_t batchsize) override
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
    ErrorType deleteGradients() override
    {
	m_ptrWeightGradient.reset();
	m_ptrBiasGradient.reset();
        return ErrorType::OK;
    }
    ErrorType loadFromFlash() override
    {
        switch (m_optimizerType) {
        case OptimizerID::SGD:
            // init weights
            m_weightPtr = std::make_unique<OptimizerSGD<T>>();
            m_weightPtr->init(m_header->weightsAmountTrainable);
            // init bias
            m_biasPtr = std::make_unique<OptimizerSGD<T>>();
            m_biasPtr->init(m_header->biasAmountTrainable);
            break;

        case OptimizerID::Momentum:
            // init weights
            m_weightPtr = std::make_unique<OptimizerMomentum<T>>();
            m_weightPtr->init(m_header->weightsAmountTrainable);
            // init bias
            m_biasPtr = std::make_unique<OptimizerMomentum<T>>();
            m_biasPtr->init(m_header->biasAmountTrainable);
            break;

        case OptimizerID::ADAM:
            // init weights
            m_weightPtr = std::make_unique<OptimizerAdam<T>>();
            m_weightPtr->init(m_header->weightsAmountTrainable);
            // init bias
            m_biasPtr = std::make_unique<OptimizerAdam<T>>();
            m_biasPtr->init(m_header->biasAmountTrainable);
            break;
        }

        // init weights and biases
        for (size_t i = 0; i < m_header->weightsAmountTrainable; i++) {
            m_weightPtr->setData(i, *(m_ptrData + (m_header->weightsTrainableOffset / sizeof(T)) + i));
        }
        for (size_t i = 0; i < m_header->biasAmountTrainable; i++) {
            m_biasPtr->setData(i, *(m_ptrData + (m_header->biasTrainableOffset / sizeof(T)) + i));
        }

        this->m_isLoaded = true;
        return ErrorType::OK;
    }

    ErrorType storeToFlash() override
    {
        // not implemented for this version
        return ErrorType::OK;
    }

    uint32_t getOutputSize() override
    {
        return m_header->dimensionOutputX;
    }

    uint32_t getInputSize() override
    {
        return m_header->dimensionInputX;
    }

private:

    const T* m_ptrData;
    Nmcm::BatchNorm1d::NeuralNetwork* m_header;
    OptimizerID m_optimizerType;


	std::unique_ptr<double[]>	m_runningMean;
	std::unique_ptr<double[]>	m_runningVar;

    // TODO fix datatypes
    T* m_normalizedInput = nullptr;
    T* m_input = nullptr;

    // flag for first run, to init running mean/var
    bool m_initRunningStats = false;

	const T* m_ptrFlashWeight = nullptr;
	const T* m_ptrFlashBias = nullptr;

	std::unique_ptr<T[]> m_ptrWeightGradient;
	std::unique_ptr<T[]> m_ptrBiasGradient;
	
	std::unique_ptr<OptimizerBase<T>> m_weightPtr;
	std::unique_ptr<OptimizerBase<T>> m_biasPtr;

    uint32_t m_timestep = 1;

};
} // namespace Edgeist

#endif // EDGEIST_BATCH_NORM_1D_H
