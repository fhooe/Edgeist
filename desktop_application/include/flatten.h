/**
 * @author David Muttenthaler
 * @brief Implements the flatten layer.
 **/

#ifndef FLATTEN_H
#define FLATTEN_H

#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {

template <typename T>
class model;

/**
 * @brief Flatten layer.
 *
 * Flattens a multi-dimensional input tensor into a 1D tensor.
 * Commonly used to connect convolutional layers with fully connected layers.
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class Flatten : public Layer<T> {
public:
    // Konstruktor: Initialisiert die ReLU-Schicht mit Zeigern auf Konfigurationsdaten und gew�hltem Optimierer
    Flatten(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {

        this->mHeader = static_cast<Neural_Network_Flatten_t*>(mPtrLayer);
    }

    // Destruktor: Gibt dynamisch allokierten Speicher frei
    ~Flatten() override = default;

    // Führt die Vorwärtspassage durch und schreibt das Ergebnis in output.
    auto forwardPass(const T* input_data, T* output_data, bool /* trainingflag */) -> ErrorType override
    {
        for (size_t i = 0; i < mHeader->dimensioninput_x; i++) {
            output_data[i] = input_data[i];
        }
        return ErrorType::ok;
    }

    // Führt die Rückwärtspassage durch und berechnet die Gradienten für den vorherigen Layer.
    auto backwardPass(const T* input_data, T* output_data) -> ErrorType override
    {
        for (size_t i = 0; i < mHeader->dimensioninput_x; i++) {
            output_data[i] = input_data[i];
        }
        return ErrorType::ok;
    }

    // Lädt die trainierbaren werte vom Flash in den SRAM
    auto loadFromFlash() -> ErrorType override
    {
        ;
        return ErrorType::ok;
    }

    // Speichert die trainierbaren werte vom SRAM in den Flash
    auto storeToFlash() -> ErrorType override
    {
        return ErrorType::ok;
    }

    auto getOutputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensionoutput_x);
    }

    auto getInputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensioninput_x);
    }

    [[nodiscard]] auto header() const -> const Neural_Network_Flatten_t&
    {
        return *mHeader;
    }

private:
    void* mPtrLayer;

    void* mPtrData;

    // Pointer to the Layer in Flash
    Neural_Network_Flatten_t* mHeader;

    // chosen optimizer
    OptimizerID mOptimizerType;
};
} // namespace Edgeist

#endif // FLATTEN_H
