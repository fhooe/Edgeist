/**
 * @author David Muttenthaler
 * @brief Implements the fully connected (dense) linear layer.
 **/

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
    // Temporär, bis typedef in "Modeltypes.h" integriert ist
    using Linear_DataType_t = T;

    // Konstruktor: Initialisiert die Linear-Schicht mit Zeigern auf Konfigurationsdaten und gewähltem Optimierer
    Linear(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
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

    // Destruktor: Gibt dynamisch allokierten Speicher frei
    ~Linear() override
    {

        // Free memory
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

    // Führt die Vorwärtspassage durch und schreibt das Ergebnis in output.
    auto forwardPass(const T* input_data, T* output_data, bool trainingflag) -> ErrorType override
    {
        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

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

        // Calc Result
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

    // Führt die Rückwärtspassage durch und berechnet die Gradienten für den vorherigen Layer.
    auto backwardPass(const T* input_data, T* output_data) -> ErrorType override
    {
        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        if (this->mInputData == nullptr) {
            return ErrorType::UnknownError;
        }

        // do Backward Pass
        T* dL_dW = new T[this->mHeader->dimensionoutput_x * this->mHeader->dimensioninput_x];
        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
            for (size_t j = 0; j < this->mHeader->dimensioninput_x; ++j) {
                dL_dW[i * this->mHeader->dimensioninput_x + j] = input_data[i] * this->mInputData[j];
            }
        }

        // 2. Gradient für Bias
        T* dL_db = new T[this->mHeader->dimensionoutput_x];
        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
            dL_db[i] = input_data[i];
        }

        // 3. Gradient für Input x (Backprop durch Layer)
        for (size_t j = 0; j < this->mHeader->dimensioninput_x; ++j) {
            output_data[j] = 0.0f;
            for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
                output_data[j] += mWeightPtr->getData((i * (mHeader->dimensioninput_x)) + j) * input_data[i];
            }
        }

        // Update Gradients
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

    // Initialisiere Gradienten, Speicher reservieren und initialisieren
    auto initGradients() -> ErrorType override
    {

        if (mPtrWeightGradient != nullptr || mPtrBiasGradient != nullptr) {
            return ErrorType::UnknownError;
        }

        // arrays dynamisch anlegen und mit 0.0 initialisieren

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

    // Lösche Gradienten, Speicher freigeben
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

    // Update Weights and biases
    auto update(uint32_t batchsize) -> ErrorType override
    {

        if (mPtrWeightGradient == nullptr || mPtrBiasGradient == nullptr) {
            return ErrorType::UnknownError;
        }

        // SGD-Update von W und b
        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
            for (size_t j = 0; j < this->mHeader->dimensioninput_x; ++j) {
                mWeightPtr->update((i * (mHeader->dimensioninput_x)) + j, mPtrWeightGradient[i * this->mHeader->dimensioninput_x + j] / batchsize, this->mModel->mLearningRate, mTimestep);
            }
            mBiasPtr->update(i, mPtrBiasGradient[i] / batchsize, this->mModel->mLearningRate, mTimestep);
        }
        mTimestep++;
        return ErrorType::ok;
    }

    // Lädt die trainierbaren werte vom Flash in den SRAM
    auto loadFromFlash() -> ErrorType override
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
            mWeightPtr->setData(i, *(static_cast<Linear_DataType_t*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(Linear_DataType_t)) + i));
        }
        for (size_t i = 0; i < mHeader->bias_amount_trainable; i++) {
            mBiasPtr->setData(i, *(static_cast<Linear_DataType_t*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(Linear_DataType_t)) + i));
        }

        this->mIsLoaded = true;
        return ErrorType::ok;
    }

    // Speichert die trainierbaren werte vom SRAM in den Flash
    auto storeToFlash() -> ErrorType override
    {
        this->mIsLoaded = false;
        // nicht implementier in dieser Version, wird erst am µController relevant
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
