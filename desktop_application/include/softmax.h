/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the softmax activation layer.
 */

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"
#include <cmath>

namespace Edgeist {
template <typename T>
class model;

/**
 * @brief Softmax activation layer.
 *
 * Converts raw logits into probabilities by exponentiating and normalizing.
 * Used as the final layer in multi-class classification tasks.
 *
 * Output is a probability distribution (sums to 1).
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class Softmax : public Layer<T> {
public:
    // Konstruktor: Initialisiert die Softmax-Schicht mit Zeigern auf Konfigurationsdaten und gew�hltem Optimierer
    Softmax(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {

        this->mHeader = static_cast<Neural_Network_Softmax_t*>(mPtrLayer);
        this->mInputData = nullptr;

        loadFromFlash();
    }

    // Destruktor: Gibt dynamisch allokierten Speicher frei
    ~Softmax() override
    {

        if (this->mInputData != nullptr) {
            delete[] this->mInputData;
            this->mInputData = nullptr;
        }
    }

    // Führt die Vorwärtspassage (Forward Pass) durch; berechnet Softmax-Ausgabe aus Eingabedaten
    // input_data: output der Vorherigen Layer
    // output_Data: output dieses Layers
    auto forwardPass(const T* input_data, T* output_data, bool trainingflag) -> ErrorType override
    {
        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        if (this->mIsLoaded != true) {
            return ErrorType::LayerNotInitialized;
        }

        // Finde den Maximalwert der Eingabe zur numerischen Stabilisierung der Exponentialfunktion
        T maxVal = input_data[0];
        for (size_t i = 1; i < this->mHeader->dimensioninput_x; i++) {
            if (input_data[i] > maxVal) {
                maxVal = input_data[i];
            }
        }

        // Berechne exponentielle Werte der um maxVal verschobenen Eingaben und summiere sie
        T sumExponents = T(0);
        T* mPtrExponents = new T[this->mHeader->dimensioninput_x];

        for (size_t i = 0; i < this->mHeader->dimensioninput_x; i++) {
            mPtrExponents[i] = std::exp(input_data[i] - maxVal);
            sumExponents += mPtrExponents[i];
        }

        if (trainingflag) {
            // Allokiere Speicher für Zwischenspeicherung der Eingabe (für Backpropagation)
            if (this->mInputData != nullptr) {
                delete[] this->mInputData;
            }
            this->mInputData = new T[this->mHeader->dimensioninput_x];

            // Normiere Exponentialwerte zur Softmax-Ausgabe
            if (this->mInputData != nullptr) {
                for (size_t i = 0; i < this->mHeader->dimensionoutput_x; i++) {
                    this->mInputData[i] = input_data[i];
                }
            }
        }
        for (size_t i = 0; i < this->mHeader->dimensionoutput_x; i++) {
            output_data[i] = mPtrExponents[i] / sumExponents;
        }

        delete[] mPtrExponents;
        mPtrExponents = nullptr;

        return ErrorType::ok;
    }

    // Führt die Rückwärtspassage (Backward Pass) durch; berechnet den Fehlergradienten der Softmax-Schicht
    // input_data: output vom forwardPass
    // output_Data: ist Gradient
    auto backwardPass(const T* input_data, T* output_data) -> ErrorType override
    {
        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        if (this->mIsLoaded != true) {
            return ErrorType::LayerNotInitialized;
        }

        // Softmax + Cross-Entropy Ableitung: p - y
        for (size_t i = 0; i < this->mHeader->dimensioninput_x; i++) {
            output_data[i] = input_data[i] - this->mModel->mPtrExpectedOutputData[i];
        }

        return ErrorType::ok;
    }

    // Lädt die trainierbaren werte vom Flash in den SRAM
    auto loadFromFlash() -> ErrorType override
    {

        this->mIsLoaded = true;
        // Nicht relevant bei ReLU, da keine Weights und Biases
        return ErrorType::ok;
    }

    // Speichert die trainierbaren werte vom SRAM in den Flash
    auto storeToFlash() -> ErrorType override
    {
        this->mIsLoaded = false;
        // nicht implementier in dieser Version, wird erst am �Controller relevant
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

    [[nodiscard]] auto header() const -> const Neural_Network_Softmax_t&
    {
        return *mHeader;
    }

private:
    void* mPtrLayer;

    void* mPtrData;

    // Pointer to the Layer in Flash
    Neural_Network_Softmax_t* mHeader;

    // chosen optimizer
    OptimizerID mOptimizerType;
};
} // namespace Edgeist

#endif // SOFTMAX_H
