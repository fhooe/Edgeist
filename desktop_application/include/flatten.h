#pragma once
#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

/**
 * @author David Muttenthaler
 * @date 25-06-2025
 *
 * @brief Flatten Layer.
 *
 * Flattens a multi-dimensional input tensor into a 1D tensor.
 * Commonly used to connect convolutional layers with fully connected layers.
 */

template <typename T>
class model;

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
    virtual ErrorType forwardPass(const T* input_data, T* output_data, bool /* trainingflag */) override
    {
        for (size_t i = 0; i < mHeader->dimensioninput_x; i++) {
            output_data[i] = input_data[i];
        }
        return ErrorType::ok;
    }

    // Führt die Rückwärtspassage durch und berechnet die Gradienten für den vorherigen Layer.
    ErrorType backwardPass(const T* input_data, T* output_data) override
    {
        for (size_t i = 0; i < mHeader->dimensioninput_x; i++) {
            output_data[i] = input_data[i];
        }
        return ErrorType::ok;
    }

    // Lädt die trainierbaren werte vom Flash in den SRAM
    ErrorType loadFromFlash() override
    {
        ;
        return ErrorType::ok;
    }

    // Speichert die trainierbaren werte vom SRAM in den Flash
    ErrorType storeToFlash() override
    {
        return ErrorType::ok;
    }

    uint32_t getOutputSize() override
    {
        return uint32_t(mHeader->dimensionoutput_x);
    }

    uint32_t getInputSize() override
    {
        return uint32_t(mHeader->dimensioninput_x);
    }

    const Neural_Network_Flatten_t& header()
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
