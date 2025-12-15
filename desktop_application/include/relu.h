/**
 * @author David Muttenthaler
 * @brief Implements the rectified linear unit (ReLU) activation layer.
 */

#ifndef RELU_H
#define RELU_H

#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {

template <typename T>
class model;

/**
 * @brief Rectified Linear Unit (ReLU) Activation Layer.
 *
 * Applies the function `f(x) = max(0, x)` element-wise.
 * Adds non-linearity to the network and prevents vanishing gradients.
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class Relu : public Layer<T> {
public:
    // Konstruktor: Initialisiert die ReLU-Schicht mit Zeigern auf Konfigurationsdaten und gewähltem Optimierer
    Relu(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {

        this->mHeader = static_cast<Neural_Network_ReLU_t*>(mPtrLayer);
        this->mInputData = nullptr;

        loadFromFlash();
    }

    // Destruktor: Gibt dynamisch allokierten Speicher frei
    ~Relu() override
    {
        if (this->mInputData != nullptr) {
            delete[] this->mInputData;
            this->mInputData = nullptr;
        }
    }

    // Führt die Vorwärtspassage durch und schreibt das Ergebnis in output.
    auto forwardPass(const T* input_data, T* output_data, bool trainingFlag) -> ErrorType override
    {
        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        if (trainingFlag) {

            if (this->mIsLoaded != true) {
                return ErrorType::LayerNotInitialized;
            }
            // get memory for training
            if (this->mInputData == nullptr) {
                this->mInputData = new T[this->mHeader->dimensioninput_x];
            }

            if (this->mInputData != nullptr) {
                for (size_t i = 0; i < this->mHeader->dimensionoutput_x; i++) {
                    this->mInputData[i] = input_data[i];
                }
            }
        }

        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; i++) {
            output_data[i] = (input_data[i] > T(0)) ? input_data[i] : T(0);
        }

        return ErrorType::ok;
    }

    // Führt die Rückwärtspassage durch und berechnet die Gradienten für den vorherigen Layer.
    auto backwardPass(const T* input_data, T* output_data) -> ErrorType override
    {

        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        if (this->mIsLoaded != true) {
            return ErrorType::LayerNotInitialized;
        }

        if (this->mInputData == nullptr) {
            return ErrorType::UnknownError;
        }

        for (uint32_t i = 0; i < this->mHeader->dimensioninput_x; i++) {
            output_data[i] = (this->mInputData[i] > T(0)) ? input_data[i] : T(0);
        }

        if (this->mInputData != nullptr) {
            delete[] this->mInputData;
            this->mInputData = nullptr;
        }

        return ErrorType::UnknownError;
    }

    // Lädt die trainierbaren werte vom Flash in den SRAM
    auto loadFromFlash() -> ErrorType override
    {

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

    [[nodiscard]] auto header() const -> const Neural_Network_ReLU_t&
    {
        return *mHeader;
    }

private:
    void* mPtrLayer;

    void* mPtrData;

    // Pointer to the Layer in Flash
    Neural_Network_ReLU_t* mHeader;

    // chosen optimizer
    OptimizerID mOptimizerType;
};
} // namespace Edgeist

#endif // RELU_H
