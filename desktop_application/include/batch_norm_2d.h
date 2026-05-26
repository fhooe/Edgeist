/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the 2D batch normalization layer
 */

#ifndef EDGEIST_BATCH_NORM_2D_H
#define EDGEIST_BATCH_NORM_2D_H

#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"
#include <cstdint>
#include <memory.h>
#include <memory>
#include <vector>
namespace Edgeist {
template <typename T>
class Model;

/**
 * @brief 2D batch normalization Layer
 *
 * Applies batch normalization to 2D input (e.g., channels in a 2D image tensor).
 * Normalizes each channel separately across the batch.
 *
 * Commonly used after Conv2d layers.
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class BatchNorm2D : public Layer<T> {
public:
    BatchNorm2D(Model<T>* model, void* headerPointer, void* dataPointer, const OptimizerID optimizerType)
        : Layer<T>(model)
        , m_ptrData(static_cast<const T*> (dataPointer))
        , m_header(static_cast<Nmcm::BatchNorm2d::NeuralNetwork*>(headerPointer))
        , m_optimizerType(optimizerType)
    {
	    m_runningMean = std::make_unique<double[]>(m_header->channelsIn);
	    m_runningVar = std::make_unique<double[]>(m_header->channelsIn);

        m_ptrFlashWeight = (m_ptrData + (m_header->weightsTrainableOffset / sizeof(T)));
        m_ptrFlashBias = (m_ptrData + (m_header->biasTrainableOffset / sizeof(T)));

        BatchNorm2D::loadFromFlash();
    }

    ErrorType forwardPass(const T* inputData, T* outputData, const bool trainingFlag) override
    {
        // Use ether data from flash or SRAM based on trainingflag
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
            m_initRunningStats = false;
        }

        const int nrOfInputs = m_header->dimensionInputX;
        const int nrOfChannels = m_header->channelsIn;
        const int elemsPerChannel = nrOfInputs / nrOfChannels;
        float const eps = 1e-5f;
	
	    std::vector<double> mean(nrOfChannels,0.0);
	    std::vector<double> var(nrOfChannels,1.0);

        for (int i = 0; i < nrOfChannels; i++) {
            mean[i] = double(0.0);
            var[i] = double(1.0);
        }

        // calc average
        for (int c = 0; c < nrOfChannels; c++) {
            for (int i = 0; i < elemsPerChannel; i++) {
                mean[c] += static_cast<double>(inputData[c * elemsPerChannel + i]);
            }
            mean[c] = mean[c] / elemsPerChannel;
        }

        // calc variance
        for (int c = 0; c < nrOfChannels; c++) {
            for (int i = 0; i < elemsPerChannel; i++) {
                double diff = static_cast<double>(inputData[c * elemsPerChannel + i]) - mean[c];
                var[c] += diff * diff;
            }

            var[c] = var[c] / elemsPerChannel;
        }

        // running_Mean
        if (!m_initRunningStats) {
            // initialize with the values from buffer
            for (int c = 0; c < nrOfChannels; c++) {
                m_runningMean[c] = ptrWeight[c + m_header->channelsIn];
                m_runningVar[c] = ptrBias[c + m_header->channelsIn];
            }
            m_initRunningStats = true;
        }

        if (trainingFlag) { // forward pass with training

            float momentum = 0.1f;
            for (int c = 0; c < nrOfChannels; c++) {
                m_runningMean[c] = (1 - momentum) * m_runningMean[c] + momentum * mean[c];
                m_runningVar[c] = (1 - momentum) * m_runningVar[c] + momentum * var[c];
            }

            m_normalizedInput = std::make_unique<T[]>(nrOfInputs);
            m_input = std::make_unique<T[]>(nrOfInputs);

            // normalize + scale + move
            for (int i = 0; i < nrOfChannels; i++) {

                T gamma = ptrWeight[i];
                T beta = ptrBias[i];

                for (int x = 0; x < nrOfInputs / nrOfChannels; x++) {
                    int index = i * (nrOfInputs / nrOfChannels) + x;

                    double norm = (double(inputData[index]) - m_runningMean[i]) / std::sqrt(m_runningVar[i] + eps);

                    // save input and normaliced input
                    m_normalizedInput[index] = norm;
                    m_input[index] = inputData[index];

                    outputData[index] = gamma * norm + beta;
                }
            }

        } else {
            // normalize + scale + move
            for (int c = 0; c < nrOfChannels; c++) {
                T gamma = ptrWeight[c];
                T beta = ptrBias[c];

                for (int x = 0; x < elemsPerChannel; x++) {
                    int index = c * elemsPerChannel + x;
                    double norm = (double(inputData[index]) - m_runningMean[c]) / std::sqrt(m_runningVar[c] + eps);
                    outputData[index] = gamma * norm + beta;
                }
            }
        }

//        delete[] mean;
//        mean = nullptr;
//        delete[] var;
//        var = nullptr;
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

        const int nrOfInputs = m_header->dimensionInputX;
        const int nrOfChannels = m_header->channelsIn;
        int ElemsPerChannel = nrOfInputs / nrOfChannels; // ElemsPerChannel
        float eps = 1e-5f;

        // alloc memory
        T* dL_dgamma = new T[nrOfChannels];
        T* dL_dbeta = new T[nrOfChannels];
        double* sum_dnorm = new double[nrOfChannels];
        double* sum_dnorm_norm = new double[nrOfChannels];

        for (int c = 0; c < nrOfChannels; ++c) {
            dL_dgamma[c] = 0.0;
            dL_dbeta[c] = 0.0;
            sum_dnorm[c] = 0.0;
            sum_dnorm_norm[c] = 0.0;
        }

        // 1. calc dL/dy, dL/db, dL/dx per channel
        for (int c = 0; c < nrOfChannels; ++c) {
            for (int i = 0; i < ElemsPerChannel; ++i) {
                int idx = c * ElemsPerChannel + i;

                double norm_input = m_normalizedInput[idx];
                double dy = static_cast<double>(inputData[idx]);

                dL_dgamma[c] += dy * norm_input;
                dL_dbeta[c] += dy;

                sum_dnorm[c] += dy * m_ptrWeight->getData(c); // dL/dx = dL/dy * y
                sum_dnorm_norm[c] += (dy * m_ptrWeight->getData(c)) * norm_input;
            }
        }

        // 2. calc dL/dx per element
        for (int c = 0; c < nrOfChannels; ++c) {
            double std_inv = 1.0 / std::sqrt(m_runningVar[c] + eps);

            for (int i = 0; i < ElemsPerChannel; ++i) {
                int idx = c * ElemsPerChannel + i;
                double dy_gamma = inputData[idx] * m_ptrWeight->getData(c);

                double term1 = ElemsPerChannel * dy_gamma;
                double term2 = sum_dnorm[c];
                double term3 = m_normalizedInput[idx] * sum_dnorm_norm[c];

                outputData[idx] = (1.0 / ElemsPerChannel) * std_inv * (term1 - term2 - term3);
            }
        }

        // 3. save gradient in puffer
        for (int c = 0; c < nrOfChannels; ++c) {
            m_ptrWeightGradient[c] += dL_dgamma[c];
            m_ptrBiasGradient[c] += dL_dbeta[c];
        }

        // cleanup
        delete[] dL_dgamma;
        delete[] dL_dbeta;
        delete[] sum_dnorm;
        delete[] sum_dnorm_norm;

	    m_input.reset();	
        m_normalizedInput.reset();

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
        //m_ptrBiasGradient = new T[sizeBias];

        for (uint32_t i = 0; i < sizeWeights; ++i) {
            m_ptrWeightGradient[i] = T(1.0);
        }

        for (uint32_t i = 0; i < sizeBias; ++i) {
            m_ptrBiasGradient[i] = T(0.0);
        }
        return ErrorType::OK;
    }

    // delete mean and variance after minibatch
    ErrorType deleteGradients() override
    {
	m_ptrWeightGradient.reset();
	m_ptrBiasGradient.reset();
        return ErrorType::OK;
    }

    ErrorType update(uint32_t batchsize) override
    {

        if (m_ptrWeightGradient == nullptr || m_ptrBiasGradient == nullptr) {
            return ErrorType::UnknownError;
        }

        // SGD-Update of W and b
        for (size_t i = 0; i < m_header->dimensionInputX; ++i) {
            m_ptrWeight->update(i, m_ptrWeightGradient[i] / batchsize, this->m_model->m_learningRate, m_timestep);
            m_ptrBias->update(i, m_ptrBiasGradient[i] / batchsize, this->m_model->m_learningRate, m_timestep);
        }
        m_timestep++;

        return ErrorType::OK;
    }

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
         //   m_ptrWeight->setData(i, *m_ptrData + (m_header->weightsTrainableOffset / sizeof(T)) + i);
        	m_ptrWeight->setData(i, m_ptrFlashWeight[i]);
        }
        for (size_t i = 0; i < m_header->biasAmountTrainable; i++) {
        //    m_ptrBias->setData(i, *m_ptrData + (m_header->biasTrainableOffset / sizeof(T)) + i);
        	m_ptrBias->setData(i, m_ptrFlashBias[i]);
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
    const T*                          m_ptrData;               // non-owning, typisiert
    Nmcm::BatchNorm2d::NeuralNetwork* m_header;                // non-owning
    OptimizerID                       m_optimizerType;
	
    //why not float?
    std::unique_ptr<double[]>          m_runningMean;
    std::unique_ptr<double[]>          m_runningVar;

    std::unique_ptr<T[]>              m_normalizedInput;
    std::unique_ptr<T[]>              m_input;

    const T*                          m_ptrFlashWeight = nullptr; // non-owning (Flash)
    const T*                          m_ptrFlashBias   = nullptr; // non-owning (Flash)

    std::unique_ptr<T[]>              m_ptrWeightGradient;
    std::unique_ptr<T[]>              m_ptrBiasGradient;

    std::unique_ptr<OptimizerBase<T>> m_ptrWeight;
    std::unique_ptr<OptimizerBase<T>> m_ptrBias;

    bool     m_initRunningStats = false;
    int      m_batchIndex       = 0;
    uint32_t m_timestep         = 1;
};
} // namespace Edgeist

#endif // EDGEIST_BATCH_NORM_2D_H
