#pragma once
#include "Modeltypes.h"
#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"
#include <limits>
#include <vector>

/**
 * @author David Muttenthaler
 * @date 25-06-2025
 *
 * @brief 1D Batch Normalization Layer.
 *
 * Applies batch normalization to 1D input (typically features across a batch).
 * Normalizes the input to zero mean and unit variance, followed by a learnable
 * scale and shift.
 *
 * Helps stabilize and accelerate training by reducing internal covariate shift.
 */

// Forward declaration of model
template <typename T>
class model;

template <typename T>
class BatchNorm1d : public Layer<T> {
public:
    typedef T BatchNorm1d_DataType_t;
    BatchNorm1d(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {
        this->mHeader = static_cast<Neural_Network_BatchNorm1d_t*>(mPtrLayer);
        loadFromFlash();

        mRunningMean = new double[this->mHeader->dimensioninput_x];
        mRunningVar = new double[this->mHeader->dimensioninput_x];

        mPtrFlashWeight = (static_cast<BatchNorm1d_DataType_t*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(BatchNorm1d_DataType_t)));
        mPtrFlashBias = (static_cast<BatchNorm1d_DataType_t*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(BatchNorm1d_DataType_t)));

        loadFromFlash();
    }

    ~BatchNorm1d() override
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

    ErrorType forwardPass(const T* input_data, T* output_data, bool trainingflag) override
    {

        // Use ether data from flash or SRAM based on trainingflag
        T* ptrWeight = nullptr;
        T* ptrBias = nullptr;
        if (trainingflag) {
            // check if layer is loaded
            if (this->mIsLoaded != true) {
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

        // Mittelwert berechnen
        for (int i = 0; i < NrOfInputs; i++) {
            mean += input_data[i];
        }
        mean = mean / NrOfInputs;

        // Varianz berechnen
        for (int i = 0; i < NrOfInputs; i++) {
            T diff = input_data[i] - mean;
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

        if (trainingflag) {
            // forward pass mit training

            float momentum = 0.1f;
            for (int i = 0; i < NrOfInputs; i++) {
                mRunningMean[i] = (1 - momentum) * mRunningMean[i] + momentum * mean;
                mRunningVar[i] = (1 - momentum) * mRunningVar[i] + momentum * var;
            }

            mNormalizedInput = new T[NrOfInputs];
            mInput = new T[NrOfInputs];

            // Normalisieren + Skalieren + Verschieben
            for (int i = 0; i < NrOfInputs; i++) {
                double norm = (double(input_data[i]) - mRunningMean[i]) / std::sqrt(mRunningVar[i] + eps);

                // input und normaliced input speichern
                mNormalizedInput[i] = norm;
                mInput[i] = input_data[i];

                T gamma = ptrWeight[i];
                T beta = ptrBias[i];
                output_data[i] = gamma * norm + beta;
            }
            // speichere Werte f�r Backwardpass mit

        } else {
            // Normalisieren + Skalieren + Verschieben
            for (int i = 0; i < NrOfInputs; i++) {
                T gamma = ptrWeight[i];
                T beta = ptrBias[i];
                double norm = (double(input_data[i]) - mRunningMean[i]) / std::sqrt(mRunningVar[i] + eps);
                output_data[i] = gamma * norm + beta;
            }
        }

        return ErrorType::ok;
    }

    ErrorType backwardPass(const T* input_data, T* output_data) override
    {
        if (input_data == nullptr || output_data == nullptr)
            return ErrorType::InvalidPointer;

        if (!mInput || !mNormalizedInput)
            return ErrorType::MissingCachedInputs;

        int NrOfInputs = this->mHeader->dimensioninput_x;
        float eps = 1e-5f;

        // Zwischenspeicher
        T* dL_dgamma = new T[NrOfInputs];
        T* dL_dbeta = new T[NrOfInputs];
        T* dL_dnorm = new T[NrOfInputs]; // dL/dx

        // Initialisieren
        for (int i = 0; i < NrOfInputs; ++i) {
            dL_dgamma[i] = input_data[i] * mNormalizedInput[i]; // dL/dy = dL/dy * x
            dL_dbeta[i] = input_data[i]; // dL/db = dL/dy
            dL_dnorm[i] = input_data[i] * mWeightPtr->getData(i); // dL/dx = dL/dy * y
        }

        // Hilfsgr��en berechnen
        double sum_dnorm = 0.0;
        double sum_dnorm_norm = 0.0;
        for (int i = 0; i < NrOfInputs; ++i) {
            sum_dnorm += dL_dnorm[i];
            sum_dnorm_norm += dL_dnorm[i] * mNormalizedInput[i];
        }

        // dL/dx berechnen
        for (int i = 0; i < NrOfInputs; ++i) {
            double std_inv = 1.0 / std::sqrt(mRunningVar[i] + eps);
            double term1 = NrOfInputs * dL_dnorm[i];
            double term2 = sum_dnorm;
            double term3 = mNormalizedInput[i] * sum_dnorm_norm;

            output_data[i] = (1.0 / NrOfInputs) * std_inv * (term1 - term2 - term3);
        }

        // Akkumulieren in Gradientenpuffer
        for (int i = 0; i < NrOfInputs; ++i) {
            mPtrWeightGradient[i] += dL_dgamma[i]; // f�r gamma
            mPtrBiasGradient[i] += dL_dbeta[i]; // f�r beta
        }

        // Aufr�umen
        delete[] dL_dgamma;
        dL_dgamma = nullptr;
        delete[] dL_dbeta;
        dL_dbeta = nullptr;
        delete[] dL_dnorm;
        dL_dnorm = nullptr;
        // output_data ist output

        delete[] mInput;
        mInput = nullptr;
        delete[] mNormalizedInput;
        mNormalizedInput = nullptr;

        return ErrorType::ok;
    }

    // Initialisiert Mittelwert und varianz vor einem mini Batch
    ErrorType initGradients() override
    {
        if (mPtrWeightGradient != nullptr || mPtrBiasGradient != nullptr) {
            return ErrorType::UnknownError;
        }

        // arrays dynamisch anlegen und mit 0.0 initialisieren

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

    // Lösche Mittlwert und varianz nach minibatch
    ErrorType deleteGradients() override
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

    ErrorType update(uint32_t batchsize) override
    {

        if (mPtrWeightGradient == nullptr || mPtrBiasGradient == nullptr) {
            return ErrorType::UnknownError;
        }

        // SGD-Update von W und b
        for (BatchNorm1d_DimensionInput_x_t i = 0; i < this->mHeader->dimensioninput_x; ++i) {

            mWeightPtr->update(i, mPtrWeightGradient[i] / batchsize, this->mModel->mLearningRate, mTimestep);

            mBiasPtr->update(i, mPtrBiasGradient[i] / batchsize, this->mModel->mLearningRate, mTimestep);
        }
        mTimestep++;

        return ErrorType::ok;
    }

    ErrorType loadFromFlash() override
    {

        switch (mOptimizerType) {
        case SGD:
            // initailisiere Weights
            mWeightPtr = new OptimizerSGD<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // initailisiere Bias
            mBiasPtr = new OptimizerSGD<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;

        case Momentum:
            // initailisiere Weights
            mWeightPtr = new OptimizerMomentum<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // initailisiere Bias
            mBiasPtr = new OptimizerMomentum<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;

        case ADAM:
            // initailisiere Weights
            mWeightPtr = new OptimizerAdam<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // initailisiere Bias
            mBiasPtr = new OptimizerAdam<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;
        }

        // Initialize Weights und Biases
        for (size_t i = 0; i < mHeader->weights_amount_trainable; i++) {
            mWeightPtr->setData(i, *(static_cast<BatchNorm1d_DataType_t*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(BatchNorm1d_DataType_t)) + i));
        }
        for (size_t i = 0; i < mHeader->bias_amount_trainable; i++) {
            mBiasPtr->setData(i, *(static_cast<BatchNorm1d_DataType_t*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(BatchNorm1d_DataType_t)) + i));
        }

        this->mIsLoaded = true;
        return ErrorType::ok;
    }

    ErrorType storeToFlash() override
    {
        // not implemented for this version
        return ErrorType::ok;
    }

    uint32_t getOutputSize() override
    {
        return this->mHeader->dimensionoutput_x;
    }

    uint32_t getInputSize() override
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

    // TODO richtige Datentypen
    T* mNormalizedInput = nullptr;
    T* mInput = nullptr;

    // flag vor first runn, to init Running mean/var
    bool mInitRunningStats = false;

    BatchNorm1d_DataType_t* mPtrFlashWeight = nullptr;
    BatchNorm1d_DataType_t* mPtrFlashBias = nullptr;

    BatchNorm1d_DataType_t* mPtrWeightGradient = nullptr;
    BatchNorm1d_DataType_t* mPtrBiasGradient = nullptr;

    OptimizerBase<T>* mWeightPtr;
    OptimizerBase<T>* mBiasPtr;

    uint32_t mTimestep = 1;
};
