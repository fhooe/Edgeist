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
    // temporary until typedef is integrated into "Modeltypes.h"
    using Linear_DataType_t = T;

    // CTor: Init the ReLU-Layer with pointers to config data and the chosen optimizer
    Linear(Model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {

        mHeader = static_cast<Neural_Network_Linear_t*>(mPtrLayer);

        mPtrWeightFrozen = static_cast<Linear_DataType_t*>(mPtrLayer) + mHeader->weights_frozen_offset * sizeof(Linear_DataType_t);
        mPtrBiasFrozen = static_cast<Linear_DataType_t*>(mPtrLayer) + mHeader->bias_frozen_offset * sizeof(Linear_DataType_t);

        mPtrFlashWeight = (static_cast<Linear_DataType_t*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(Linear_DataType_t)));
        mPtrFlashBias = (static_cast<Linear_DataType_t*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(Linear_DataType_t)));

        loadFromFlash();
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
    auto forwardPass(const T* input_data, T* output_data, bool trainingflag) -> ErrorType override
    {
        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        // use ether data from flash or SRAM based on trainingflag
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
        }

        if (trainingflag) {
            // get memory for training
            if (this->mInputData == nullptr) {
                this->mInputData = new T[this->mHeader->dimensioninput_x];
            }
            for (uint32_t i = 0; i < mHeader->dimensioninput_x; i++) {
                this->mInputData[i] = input_data[i];
            }
        }

        // calc result
        // step through outputs
        for (uint32_t b = 0; b < mHeader->dimensionoutput_x; b++) {

            Linear_DataType_t temp = Linear_DataType_t(0);
            for (uint32_t i = 0; i < mHeader->dimensioninput_x; i++) {
                temp += input_data[i] * ptrWeight[(b * (mHeader->dimensioninput_x)) + i];
            }

            temp = ptrBias[b] + temp;
            output_data[b] = temp;
        }

        return ErrorType::ok;
    }

    // executes the backward pass and calculates the gradient for the layer before
    auto backwardPass(const T* input_data, T* output_data) -> ErrorType override
    {
        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        if (this->mInputData == nullptr) {
            return ErrorType::UnknownError;
        }

        // do backward pass
        T* dL_dW = new T[this->mHeader->dimensionoutput_x * this->mHeader->dimensioninput_x];
        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
            for (size_t j = 0; j < this->mHeader->dimensioninput_x; ++j) {
                dL_dW[i * this->mHeader->dimensioninput_x + j] = input_data[i] * this->mInputData[j];
            }
        }

        // 2. gradient for bias
        T* dL_db = new T[this->mHeader->dimensionoutput_x];
        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
            dL_db[i] = input_data[i];
        }

        // 3. gradient for input x (backprop through layer)
        for (size_t j = 0; j < this->mHeader->dimensioninput_x; ++j) {
            output_data[j] = 0.0f;
            for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
                output_data[j] += mWeightPtr->getData((i * (mHeader->dimensioninput_x)) + j) * input_data[i];
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
            mWeightPtr->setData(i, *(static_cast<Linear_DataType_t*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(Linear_DataType_t)) + i));
        }
        for (size_t i = 0; i < mHeader->bias_amount_trainable; i++) {
            mBiasPtr->setData(i, *(static_cast<Linear_DataType_t*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(Linear_DataType_t)) + i));
        }

        this->mIsLoaded = true;
        return ErrorType::ok;
    }

    // Stores the trainable values ​​from SRAM to Flash
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

    Linear_DataType_t* mPtrWeightFrozen = nullptr;
    Linear_DataType_t* mPtrBiasFrozen = nullptr;

    Linear_DataType_t* mPtrFlashWeight = nullptr;
    Linear_DataType_t* mPtrFlashBias = nullptr;

    Linear_DataType_t* mPtrWeightGradient = nullptr;
    Linear_DataType_t* mPtrBiasGradient = nullptr;

    // vector with the trainabel Weigths in SRAM
    OptimizerBase<T>* mWeightPtr;

    OptimizerBase<T>* mBiasPtr;

    uint32_t mTimestep = 1;
};
} // namespace Edgeist

#endif // LINEAR_H
