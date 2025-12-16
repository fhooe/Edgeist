/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the 2D batch normalization layer.
 */

#ifndef BATCH_NORM_2D_H
#define BATCH_NORM_2D_H

#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {
template <typename T>
class Model;

/**
 * @brief 2D batch normalization Layer.
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
        , mPtrLayer(headerPointer)
        , mPtrData(dataPointer)
        , mHeader(static_cast<Neural_Network_BatchNorm2d_t*>(mPtrLayer))
        , mOptimizerType(optimizerType)
    {
        mRunningMean = new double[this->mHeader->channelsin];
        mRunningVar = new double[this->mHeader->channelsin];

        mPtrFlashWeight = (static_cast<T*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(T)));
        mPtrFlashBias = (static_cast<T*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(T)));

        BatchNorm2D::loadFromFlash();
    }

    ~BatchNorm2D() override
    {
        delete[] mWeightPtr;
        mWeightPtr = nullptr;
        delete[] mBiasPtr;
        mBiasPtr = nullptr;

        delete[] mRunningMean;
        mRunningMean = nullptr;
        delete[] mRunningVar;
        mRunningVar = nullptr;
    }

    auto forwardPass(const T* inputData, T* outputData, const bool trainingFlag) -> ErrorType override
    {
        // Use ether data from flash or SRAM based on trainingflag
        T* ptrWeight = nullptr;
        T* ptrBias = nullptr;
        if (trainingFlag) {
            // check if layer is loaded
            if (!this->mIsLoaded) {
                return ErrorType::LayerNotInitialized;
            }
            ptrWeight = mWeightPtr->mData;
            ptrBias = mBiasPtr->mData;
        } else {
            ptrWeight = mPtrFlashWeight;
            ptrBias = mPtrFlashBias;
            mInitRunningStats = false;
        }

        int NrOfInputs = this->mHeader->dimensioninput_x;
        int NrOfChannels = this->mHeader->channelsin;
        int ElemsPerChannel = NrOfInputs / NrOfChannels;
        float const eps = 1e-5f;

        double* mean = new double[NrOfChannels];
        double* var = new double[NrOfChannels];
        for (int i = 0; i < NrOfChannels; i++) {
            mean[i] = double(0.0);
            var[i] = double(1.0);
        }

        // calc average
        for (int c = 0; c < NrOfChannels; c++) {
            for (int i = 0; i < ElemsPerChannel; i++) {
                mean[c] += static_cast<double>(inputData[c * ElemsPerChannel + i]);
            }
            mean[c] = mean[c] / ElemsPerChannel;
        }

        // calc variance
        for (int c = 0; c < NrOfChannels; c++) {
            for (int i = 0; i < ElemsPerChannel; i++) {
                double diff = static_cast<double>(inputData[c * ElemsPerChannel + i]) - mean[c];
                var[c] += diff * diff;
            }

            var[c] = var[c] / ElemsPerChannel;
        }

        // running_Mean
        if (!mInitRunningStats) {
            // initialize with the values from buffer
            for (int c = 0; c < NrOfChannels; c++) {
                mRunningMean[c] = ptrWeight[c + this->mHeader->channelsin];
                mRunningVar[c] = ptrBias[c + this->mHeader->channelsin];
            }
            mInitRunningStats = true;
        }

        if (trainingFlag) { // forward pass with training

            float momentum = 0.1f;
            for (int c = 0; c < NrOfChannels; c++) {
                mRunningMean[c] = (1 - momentum) * mRunningMean[c] + momentum * mean[c];
                mRunningVar[c] = (1 - momentum) * mRunningVar[c] + momentum * var[c];
            }

            mNormalizedInput = new T[NrOfInputs];
            mInput = new T[NrOfInputs];
            // normalize + scale + move
            for (int i = 0; i < NrOfChannels; i++) {

                T gamma = ptrWeight[i];
                T beta = ptrBias[i];

                for (int x = 0; x < NrOfInputs / NrOfChannels; x++) {
                    int index = i * (NrOfInputs / NrOfChannels) + x;

                    double norm = (double(inputData[index]) - mRunningMean[i]) / std::sqrt(mRunningVar[i] + eps);

                    // save input and normaliced input
                    mNormalizedInput[index] = norm;
                    mInput[index] = inputData[index];

                    outputData[index] = gamma * norm + beta;
                }
            }

        } else {
            // normalize + scale + move
            for (int c = 0; c < NrOfChannels; c++) {
                T gamma = ptrWeight[c];
                T beta = ptrBias[c];

                for (int x = 0; x < ElemsPerChannel; x++) {
                    int index = c * ElemsPerChannel + x;
                    double norm = (double(inputData[index]) - mRunningMean[c]) / std::sqrt(mRunningVar[c] + eps);
                    outputData[index] = gamma * norm + beta;
                }
            }
        }

        delete[] mean;
        mean = nullptr;
        delete[] var;
        var = nullptr;
        return ErrorType::ok;
    }

    auto backwardPass(const T* inputData, T* outputData) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::InvalidPointer;
        }

        if (mInput == nullptr || mNormalizedInput == nullptr) {
            return ErrorType::MissingCachedInputs;
        }

        int NrOfInputs = this->mHeader->dimensioninput_x;
        int NrOfChannels = this->mHeader->channelsin;
        int ElemsPerChannel = NrOfInputs / NrOfChannels; // ElemsPerChannel
        float eps = 1e-5f;

        // alloc memory
        T* dL_dgamma = new T[NrOfChannels];
        T* dL_dbeta = new T[NrOfChannels];
        double* sum_dnorm = new double[NrOfChannels];
        double* sum_dnorm_norm = new double[NrOfChannels];

        for (int c = 0; c < NrOfChannels; ++c) {
            dL_dgamma[c] = 0.0;
            dL_dbeta[c] = 0.0;
            sum_dnorm[c] = 0.0;
            sum_dnorm_norm[c] = 0.0;
        }

        // 1. calc dL/dy, dL/db, dL/dx per channel
        for (int c = 0; c < NrOfChannels; ++c) {
            for (int i = 0; i < ElemsPerChannel; ++i) {
                int idx = c * ElemsPerChannel + i;

                double norm_input = mNormalizedInput[idx];
                double dy = static_cast<double>(inputData[idx]);

                dL_dgamma[c] += dy * norm_input;
                dL_dbeta[c] += dy;

                sum_dnorm[c] += dy * mWeightPtr->getData(c); // dL/dx = dL/dy * y
                sum_dnorm_norm[c] += (dy * mWeightPtr->getData(c)) * norm_input;
            }
        }

        // 2. calc dL/dx per element
        for (int c = 0; c < NrOfChannels; ++c) {
            double std_inv = 1.0 / std::sqrt(mRunningVar[c] + eps);

            for (int i = 0; i < ElemsPerChannel; ++i) {
                int idx = c * ElemsPerChannel + i;
                double dy_gamma = inputData[idx] * mWeightPtr->getData(c);

                double term1 = ElemsPerChannel * dy_gamma;
                double term2 = sum_dnorm[c];
                double term3 = mNormalizedInput[idx] * sum_dnorm_norm[c];

                outputData[idx] = (1.0 / ElemsPerChannel) * std_inv * (term1 - term2 - term3);
            }
        }

        // 3. save gradient in puffer
        for (int c = 0; c < NrOfChannels; ++c) {
            mPtrWeightGradient[c] += dL_dgamma[c];
            mPtrBiasGradient[c] += dL_dbeta[c];
        }

        // cleanup
        delete[] dL_dgamma;
        delete[] dL_dbeta;
        delete[] sum_dnorm;
        delete[] sum_dnorm_norm;

        delete[] mInput;
        mInput = nullptr;
        delete[] mNormalizedInput;
        mNormalizedInput = nullptr;

        return ErrorType::ok;
    }

    // initializes mean and variance before a mini batch
    auto initGradients() -> ErrorType override
    {
        if (mPtrWeightGradient != nullptr || mPtrBiasGradient != nullptr) {
            return ErrorType::UnknownError;
        }

        // create arrays dynamically and initialize them with 0.0
        uint32_t sizeWeights = this->mHeader->dimensioninput_x;
        uint32_t sizeBias = this->mHeader->dimensioninput_x;

        mPtrWeightGradient = new T[sizeWeights];
        mPtrBiasGradient = new T[sizeBias];

        for (uint32_t i = 0; i < sizeWeights; ++i) {
            mPtrWeightGradient[i] = T(1.0);
        }

        for (uint32_t i = 0; i < sizeBias; ++i) {
            mPtrBiasGradient[i] = T(0.0);
        }
        return ErrorType::ok;
    }

    // delete mean and variance after minibatch
    auto deleteGradients() -> ErrorType override
    {
        if (mPtrWeightGradient != nullptr) {
            delete[] mPtrWeightGradient;
            mPtrWeightGradient = nullptr;
        }

        if (mPtrBiasGradient != nullptr) {
            delete[] mPtrBiasGradient;
            mPtrBiasGradient = nullptr;
        }
        return ErrorType::ok;
    }

    auto update(uint32_t batchsize) -> ErrorType override
    {

        if (mPtrWeightGradient == nullptr || mPtrBiasGradient == nullptr) {
            return ErrorType::UnknownError;
        }

        // SGD-Update of W and b
        for (size_t i = 0; i < this->mHeader->dimensioninput_x; ++i) {

            mWeightPtr->update(i, mPtrWeightGradient[i] / batchsize, this->mModel->mLearningRate, mTimestep);

            mBiasPtr->update(i, mPtrBiasGradient[i] / batchsize, this->mModel->mLearningRate, mTimestep);
        }
        mTimestep++;

        return ErrorType::ok;
    }

    auto loadFromFlash() -> ErrorType override
    {
        switch (mOptimizerType) {
        case OptimizerID::SGD:
            // init weights
            mWeightPtr = new OptimizerSGD<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // init bias
            mBiasPtr = new OptimizerSGD<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;

        case OptimizerID::Momentum:
            // init weights
            mWeightPtr = new OptimizerMomentum<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // init bias
            mBiasPtr = new OptimizerMomentum<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;

        case OptimizerID::ADAM:
            // init weights
            mWeightPtr = new OptimizerAdam<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // init bias
            mBiasPtr = new OptimizerAdam<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;
        }

        // init weights and biases
        for (size_t i = 0; i < mHeader->weights_amount_trainable; i++) {
            mWeightPtr->setData(i, *(static_cast<T*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(T)) + i));
        }
        for (size_t i = 0; i < mHeader->bias_amount_trainable; i++) {
            mBiasPtr->setData(i, *(static_cast<T*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(T)) + i));
        }

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
        return this->mHeader->dimensionoutput_x;
    }

    auto getInputSize() -> uint32_t override
    {
        return this->mHeader->dimensioninput_x;
    }

private:
    void* mPtrLayer;
    void* mPtrData;
    Neural_Network_BatchNorm2d_t* mHeader;
    OptimizerID mOptimizerType;

    double* mRunningMean = nullptr;
    double* mRunningVar = nullptr;

    int mBatchIndex = 0;

    // TODO fix datatypes
    T* mNormalizedInput = nullptr;
    T* mInput = nullptr;

    // flag for first run, to init running mean/var
    bool mInitRunningStats = false;

    T* mPtrFlashWeight = nullptr;
    T* mPtrFlashBias = nullptr;

    T* mPtrWeightGradient = nullptr;
    T* mPtrBiasGradient = nullptr;

    OptimizerBase<T>* mWeightPtr;
    OptimizerBase<T>* mBiasPtr;

    uint32_t mTimestep = 1;
};
} // namespace Edgeist

#endif // BATCH_NORM_2D_H
