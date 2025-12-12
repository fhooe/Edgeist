/**
 * @author David Muttenthaler
 * @brief Implements the 2D batch normalization layer.
 **/

#ifndef BATCH_NORM_2D_H
#define BATCH_NORM_2D_H

#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {
template <typename T>
class model;

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
class BatchNorm2d : public Layer<T> {
public:
    typedef T BatchNorm2d_DataType_t;
    BatchNorm2d(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {
        this->mHeader = static_cast<Neural_Network_BatchNorm2d_t*>(mPtrLayer);

        mRunningMean = new double[this->mHeader->channelsin];
        mRunningVar = new double[this->mHeader->channelsin];

        mPtrFlashWeight = (static_cast<BatchNorm2d_DataType_t*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(BatchNorm2d_DataType_t)));
        mPtrFlashBias = (static_cast<BatchNorm2d_DataType_t*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(BatchNorm2d_DataType_t)));

        loadFromFlash();
    }

    ~BatchNorm2d() override
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
        int NrOfChannels = this->mHeader->channelsin;
        int ElemsPerChannel = NrOfInputs / NrOfChannels;
        float const eps = 1e-5f;

        double* mean = new double[NrOfChannels];
        double* var = new double[NrOfChannels];
        for (int i = 0; i < NrOfChannels; i++) {
            mean[i] = double(0.0);
            var[i] = double(1.0);
        }

        // Mittelwert berechnen
        for (int c = 0; c < NrOfChannels; c++) {
            for (int i = 0; i < ElemsPerChannel; i++) {
                mean[c] += static_cast<double>(input_data[c * ElemsPerChannel + i]);
            }
            mean[c] = mean[c] / ElemsPerChannel;
        }

        // Varianz berechnen
        for (int c = 0; c < NrOfChannels; c++) {
            for (int i = 0; i < ElemsPerChannel; i++) {
                double diff = static_cast<double>(input_data[c * ElemsPerChannel + i]) - mean[c];
                var[c] += diff * diff;
            }

            var[c] = var[c] / ElemsPerChannel;
        }

        // Running_Mean
        if (!mInitRunningStats) {
            // initialize with the values from buffer
            for (int c = 0; c < NrOfChannels; c++) {
                mRunningMean[c] = ptrWeight[c + this->mHeader->channelsin];
                mRunningVar[c] = ptrBias[c + this->mHeader->channelsin];
            }
            mInitRunningStats = true;
        }

        if (trainingflag) { // forward pass mit training

            float momentum = 0.1f;
            for (int c = 0; c < NrOfChannels; c++) {
                mRunningMean[c] = (1 - momentum) * mRunningMean[c] + momentum * mean[c];
                mRunningVar[c] = (1 - momentum) * mRunningVar[c] + momentum * var[c];
            }

            mNormalizedInput = new T[NrOfInputs];
            mInput = new T[NrOfInputs];
            // Normalisieren + Skalieren + Verschieben
            for (int i = 0; i < NrOfChannels; i++) {

                T gamma = ptrWeight[i];
                T beta = ptrBias[i];

                for (int x = 0; x < NrOfInputs / NrOfChannels; x++) {
                    int index = i * (NrOfInputs / NrOfChannels) + x;

                    double norm = (double(input_data[index]) - mRunningMean[i]) / std::sqrt(mRunningVar[i] + eps);

                    // input und normaliced input speichern
                    mNormalizedInput[index] = norm;
                    mInput[index] = input_data[index];

                    output_data[index] = gamma * norm + beta;
                }
            }

        } else {
            // Normalisieren + Skalieren + Verschieben
            for (int c = 0; c < NrOfChannels; c++) {
                T gamma = ptrWeight[c];
                T beta = ptrBias[c];

                for (int x = 0; x < ElemsPerChannel; x++) {
                    int index = c * ElemsPerChannel + x;
                    double norm = (double(input_data[index]) - mRunningMean[c]) / std::sqrt(mRunningVar[c] + eps);
                    output_data[index] = gamma * norm + beta;
                }
            }
        }

        delete[] mean;
        mean = nullptr;
        delete[] var;
        var = nullptr;
        return ErrorType::ok;
    }

    ErrorType backwardPass(const T* input_data, T* output_data) override
    {
        if (!input_data || !output_data)
            return ErrorType::InvalidPointer;

        if (!mInput || !mNormalizedInput)
            return ErrorType::MissingCachedInputs;

        int NrOfInputs = this->mHeader->dimensioninput_x;
        int NrOfChannels = this->mHeader->channelsin;
        int ElemsPerChannel = NrOfInputs / NrOfChannels; // ElemsPerChannel
        float eps = 1e-5f;

        // Speicher allokieren
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

        // 1. Berechne dL/dy, dL/db, dL/dx pro Channel
        for (int c = 0; c < NrOfChannels; ++c) {
            for (int i = 0; i < ElemsPerChannel; ++i) {
                int idx = c * ElemsPerChannel + i;

                double norm_input = mNormalizedInput[idx];
                double dy = static_cast<double>(input_data[idx]);

                dL_dgamma[c] += dy * norm_input;
                dL_dbeta[c] += dy;

                sum_dnorm[c] += dy * mWeightPtr->getData(c); // dL/dx = dL/dy * y
                sum_dnorm_norm[c] += (dy * mWeightPtr->getData(c)) * norm_input;
            }
        }

        // 2. Berechne dL/dx pro Element
        for (int c = 0; c < NrOfChannels; ++c) {
            double std_inv = 1.0 / std::sqrt(mRunningVar[c] + eps);

            for (int i = 0; i < ElemsPerChannel; ++i) {
                int idx = c * ElemsPerChannel + i;
                double dy_gamma = input_data[idx] * mWeightPtr->getData(c);

                double term1 = ElemsPerChannel * dy_gamma;
                double term2 = sum_dnorm[c];
                double term3 = mNormalizedInput[idx] * sum_dnorm_norm[c];

                output_data[idx] = (1.0 / ElemsPerChannel) * std_inv * (term1 - term2 - term3);
            }
        }

        // 3. Gradienten in Puffer schreiben
        for (int c = 0; c < NrOfChannels; ++c) {
            mPtrWeightGradient[c] += dL_dgamma[c];
            mPtrBiasGradient[c] += dL_dbeta[c];
        }

        // Aufräumen
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
        for (size_t i = 0; i < this->mHeader->dimensioninput_x; ++i) {

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
            mWeightPtr->setData(i, *(static_cast<BatchNorm2d_DataType_t*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(BatchNorm2d_DataType_t)) + i));
        }
        for (size_t i = 0; i < mHeader->bias_amount_trainable; i++) {
            mBiasPtr->setData(i, *(static_cast<BatchNorm2d_DataType_t*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(BatchNorm2d_DataType_t)) + i));
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
    Neural_Network_BatchNorm2d_t* mHeader;
    OptimizerID mOptimizerType;

    double* mRunningMean = nullptr;
    double* mRunningVar = nullptr;

    int mBatchIndex = 0;

    // TODO richtige Datentypen
    T* mNormalizedInput = nullptr;
    T* mInput = nullptr;

    // flag vor first runn, to init Running mean/var
    bool mInitRunningStats = false;

    BatchNorm2d_DataType_t* mPtrFlashWeight = nullptr;
    BatchNorm2d_DataType_t* mPtrFlashBias = nullptr;

    BatchNorm2d_DataType_t* mPtrWeightGradient = nullptr;
    BatchNorm2d_DataType_t* mPtrBiasGradient = nullptr;

    OptimizerBase<T>* mWeightPtr;
    OptimizerBase<T>* mBiasPtr;

    uint32_t mTimestep = 1;
};
} // namespace Edgeist

#endif // BATCH_NORM_2D_H
