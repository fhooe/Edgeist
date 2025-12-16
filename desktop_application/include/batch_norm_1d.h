/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the 1D batch normalization layer.
 */

#ifndef BATCH_NORM_1D_H
#define BATCH_NORM_1D_H

#include "Modeltypes.h"
#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {
template <typename T>
class Model;

/**
 * @brief 1D batch normalization layer.
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
        , mPtrLayer(headerPointer)
        , mPtrData(dataPointer)
        , mHeader(static_cast<Neural_Network_BatchNorm1d_t*>(mPtrLayer))
        , mOptimizerType(optimizerType)
    {
        BatchNorm1D::loadFromFlash();

        mRunningMean = new double[this->mHeader->dimensioninput_x];
        mRunningVar = new double[this->mHeader->dimensioninput_x];

        mPtrFlashWeight = (static_cast<T*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(T)));
        mPtrFlashBias = (static_cast<T*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(T)));

        BatchNorm1D::loadFromFlash();
    }

    ~BatchNorm1D() override
    {
        delete[] mWeightPtr;
        mWeightPtr = nullptr;
        delete[] mBiasPtr;
        mBiasPtr = nullptr;

        delete mRunningMean;
        mRunningMean = nullptr;
        delete mRunningVar;
        mRunningVar = nullptr;
    }

    auto forwardPass(const T* inputData, T* outputData, bool trainingFlag) -> ErrorType override
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

        float eps = 1e-5f;

        double mean = double(0.0);
        double var = double(1.0);

        // calculate average
        for (int i = 0; i < NrOfInputs; i++) {
            mean += inputData[i];
        }
        mean = mean / NrOfInputs;

        // calculate variance
        for (int i = 0; i < NrOfInputs; i++) {
            T diff = inputData[i] - mean;
            var += diff * diff;
        }
        var = var / NrOfInputs;

        if (mInitRunningStats == false) {
            // initialize with the values from buffer
            for (int i = 0; i < NrOfInputs; i++) {
                mRunningMean[i] = ptrWeight[i + NrOfInputs];
                mRunningVar[i] = ptrBias[i + NrOfInputs];
            }
            mInitRunningStats = true;
        }

        if (trainingFlag) {
            // forward pass with training
            float momentum = 0.1f;
            for (int i = 0; i < NrOfInputs; i++) {
                mRunningMean[i] = (1 - momentum) * mRunningMean[i] + momentum * mean;
                mRunningVar[i] = (1 - momentum) * mRunningVar[i] + momentum * var;
            }

            mNormalizedInput = new T[NrOfInputs];
            mInput = new T[NrOfInputs];

            // normalize + scale + move
            for (int i = 0; i < NrOfInputs; i++) {
                double norm = (double(inputData[i]) - mRunningMean[i]) / std::sqrt(mRunningVar[i] + eps);

                // save input and normaliced input
                mNormalizedInput[i] = norm;
                mInput[i] = inputData[i];

                T gamma = ptrWeight[i];
                T beta = ptrBias[i];
                outputData[i] = gamma * norm + beta;
            }
            // store values for backwardpass

        } else {
            // normalize + scale + move
            for (int i = 0; i < NrOfInputs; i++) {
                T gamma = ptrWeight[i];
                T beta = ptrBias[i];
                double norm = (double(inputData[i]) - mRunningMean[i]) / std::sqrt(mRunningVar[i] + eps);
                outputData[i] = gamma * norm + beta;
            }
        }

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
        float eps = 1e-5f;

        // cache
        T* dL_dgamma = new T[NrOfInputs];
        T* dL_dbeta = new T[NrOfInputs];
        T* dL_dnorm = new T[NrOfInputs]; // dL/dx

        // init
        for (int i = 0; i < NrOfInputs; ++i) {
            dL_dgamma[i] = inputData[i] * mNormalizedInput[i]; // dL/dy = dL/dy * x
            dL_dbeta[i] = inputData[i]; // dL/db = dL/dy
            dL_dnorm[i] = inputData[i] * mWeightPtr->getData(i); // dL/dx = dL/dy * y
        }

        // calculate helper values
        double sum_dnorm = 0.0;
        double sum_dnorm_norm = 0.0;
        for (int i = 0; i < NrOfInputs; ++i) {
            sum_dnorm += dL_dnorm[i];
            sum_dnorm_norm += dL_dnorm[i] * mNormalizedInput[i];
        }

        // calculate dL/dx
        for (int i = 0; i < NrOfInputs; ++i) {
            double std_inv = 1.0 / std::sqrt(mRunningVar[i] + eps);
            double term1 = NrOfInputs * dL_dnorm[i];
            double term2 = sum_dnorm;
            double term3 = mNormalizedInput[i] * sum_dnorm_norm;

            outputData[i] = (1.0 / NrOfInputs) * std_inv * (term1 - term2 - term3);
        }

        // accumulate within gradient buffer
        for (int i = 0; i < NrOfInputs; ++i) {
            mPtrWeightGradient[i] += dL_dgamma[i]; // for gamma
            mPtrBiasGradient[i] += dL_dbeta[i]; // for beta
        }

        // cleanup
        delete[] dL_dgamma;
        dL_dgamma = nullptr;
        delete[] dL_dbeta;
        dL_dbeta = nullptr;
        delete[] dL_dnorm;
        dL_dnorm = nullptr;
        // output_data is output

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
        for (BatchNorm1d_DimensionInput_x_t i = 0; i < this->mHeader->dimensioninput_x; ++i) {

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
    Neural_Network_BatchNorm1d_t* mHeader;
    OptimizerID mOptimizerType;

    double* mRunningMean = nullptr;
    double* mRunningVar = nullptr;

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

#endif // BATCH_NORM_1D_H
