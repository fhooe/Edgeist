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
        , mPtrLayer(headerPointer)
        , mPtrData(dataPointer)
        , mHeader(static_cast<Neural_Network_Linear_t*>(mPtrLayer))
        , mOptimizerType(optimizerType)
    {
        mPtrWeightFrozen = static_cast<T*>(mPtrLayer) + mHeader->weights_frozen_offset * sizeof(T);
        mPtrBiasFrozen = static_cast<T*>(mPtrLayer) + mHeader->bias_frozen_offset * sizeof(T);

        mPtrFlashWeight = (static_cast<T*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(T)));
        mPtrFlashBias = (static_cast<T*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(T)));

        Linear::loadFromFlash();
    }

    // DTor: frees dynamically allocated memory
    ~Linear() override
    {
        // free memory
        if (mWeightPtr != nullptr) {
            delete[] mWeightPtr;
            mWeightPtr = nullptr;
        }

        if (mBiasPtr != nullptr) {
            delete[] mBiasPtr;
            mBiasPtr = nullptr;
        }

        if (this->mInputData != nullptr) {
            delete[] this->mInputData;
            this->mInputData = nullptr;
        }

        if (mPtrWeightGradient != nullptr) {
            delete[] mPtrWeightGradient;
            mPtrWeightGradient = nullptr;
        }

        if (mPtrBiasGradient != nullptr) {
            delete[] mPtrBiasGradient;
            mPtrBiasGradient = nullptr;
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
            if (!this->mIsLoaded) {
                return ErrorType::LayerNotInitialized;
            }
            ptrWeight = mWeightPtr->mData;
            ptrBias = mBiasPtr->mData;
        } else {
            ptrWeight = mPtrFlashWeight;
            ptrBias = mPtrFlashBias;
        }

        if (trainingFlag) {
            // get memory for training
            if (this->mInputData == nullptr) {
                this->mInputData = new T[this->mHeader->dimensioninput_x];
            }
            for (uint32_t i = 0; i < mHeader->dimensioninput_x; i++) {
                this->mInputData[i] = inputData[i];
            }
        }

        // calc result
        // step through outputs
        for (uint32_t b = 0; b < mHeader->dimensionoutput_x; b++) {
            auto temp = T(0);
            for (uint32_t i = 0; i < mHeader->dimensioninput_x; i++) {
                temp += inputData[i] * ptrWeight[(b * (mHeader->dimensioninput_x)) + i];
            }

            temp = ptrBias[b] + temp;
            outputData[b] = temp;
        }

        return ErrorType::ok;
    }

    // executes the backward pass and calculates the gradient for the layer before
    auto backwardPass(const T* inputData, T* outputData) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }

        if (this->mInputData == nullptr) {
            return ErrorType::UnknownError;
        }

        // do backward pass
        T* dL_dW = new T[this->mHeader->dimensionoutput_x * this->mHeader->dimensioninput_x];
        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
            for (size_t j = 0; j < this->mHeader->dimensioninput_x; ++j) {
                dL_dW[i * this->mHeader->dimensioninput_x + j] = inputData[i] * this->mInputData[j];
            }
        }

        // 2. gradient for bias
        T* dL_db = new T[this->mHeader->dimensionoutput_x];
        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
            dL_db[i] = inputData[i];
        }

        // 3. gradient for input x (backprop through layer)
        for (size_t j = 0; j < this->mHeader->dimensioninput_x; ++j) {
            outputData[j] = 0.0f;
            for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
                outputData[j] += mWeightPtr->getData((i * (mHeader->dimensioninput_x)) + j) * inputData[i];
            }
        }

        // update gradients
        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
            for (size_t j = 0; j < this->mHeader->dimensioninput_x; ++j) {
                mPtrWeightGradient[i * this->mHeader->dimensioninput_x + j] += dL_dW[i * this->mHeader->dimensioninput_x + j];
            }
            mPtrBiasGradient[i] += dL_db[i];
        }

        if (this->mInputData != nullptr) {
            delete[] this->mInputData;
            this->mInputData = nullptr;
        }

        delete[] dL_dW;
        dL_dW = nullptr;
        delete[] dL_db;
        dL_db = nullptr;

        return ErrorType::ok;
    }

    // init gradient, reserve and init memory
    auto initGradients() -> ErrorType override
    {
        if (mPtrWeightGradient != nullptr || mPtrBiasGradient != nullptr) {
            return ErrorType::UnknownError;
        }

        // create arrays dynamically and initialize them with 0.0
        uint32_t sizeWeights = this->mHeader->dimensionoutput_x * this->mHeader->dimensioninput_x;
        uint32_t sizeBias = this->mHeader->dimensionoutput_x;

        mPtrWeightGradient = new T[sizeWeights];
        mPtrBiasGradient = new T[sizeBias];

        for (uint32_t i = 0; i < sizeWeights; ++i) {
            mPtrWeightGradient[i] = T(0.0);
        }

        for (uint32_t i = 0; i < sizeBias; ++i) {
            mPtrBiasGradient[i] = T(0.0);
        }

        return ErrorType::ok;
    }

    // delete gradient and free memory
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

    // update weights and biases
    auto update(uint32_t batchsize) -> ErrorType override
    {
        if (mPtrWeightGradient == nullptr || mPtrBiasGradient == nullptr) {
            return ErrorType::UnknownError;
        }

        // SGD-Update of W and b
        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
            for (size_t j = 0; j < this->mHeader->dimensioninput_x; ++j) {
                mWeightPtr->update((i * (mHeader->dimensioninput_x)) + j, mPtrWeightGradient[i * this->mHeader->dimensioninput_x + j] / batchsize, this->mModel->mLearningRate, mTimestep);
            }
            mBiasPtr->update(i, mPtrBiasGradient[i] / batchsize, this->mModel->mLearningRate, mTimestep);
        }
        mTimestep++;
        return ErrorType::ok;
    }

    // Loads the trainable values from Flash into SRAM
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

    // Stores the trainable values from SRAM to Flash
    auto storeToFlash() -> ErrorType override
    {
        this->mIsLoaded = false;
        // Not implemented in this version, will only become relevant on the uController
        return ErrorType::UnknownError;
    }

    auto getOutputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensionoutput_x);
    }

    auto getInputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensioninput_x);
    }

    [[nodiscard]] auto header() const -> const Neural_Network_Linear_t&
    {
        return *mHeader;
    }

private:
    void* mPtrLayer;

    void* mPtrData;

    Neural_Network_Linear_t* mHeader;

    // chosen optimizer
    OptimizerID mOptimizerType;

    T* mPtrWeightFrozen = nullptr;
    T* mPtrBiasFrozen = nullptr;

    T* mPtrFlashWeight = nullptr;
    T* mPtrFlashBias = nullptr;

    T* mPtrWeightGradient = nullptr;
    T* mPtrBiasGradient = nullptr;

    // vector with the trainable weights in SRAM
    OptimizerBase<T>* mWeightPtr;

    OptimizerBase<T>* mBiasPtr;

    uint32_t mTimestep = 1;
};
} // namespace Edgeist

#endif // LINEAR_H
