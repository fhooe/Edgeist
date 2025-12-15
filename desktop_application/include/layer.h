/**
 * @author David Muttenthaler
 * @brief Implements the base class for all neural network layers.
 **/

#ifndef LAYER_H
#define LAYER_H

#include "nmcf_error_types.h"
#include "object.h"

namespace Edgeist {
template <typename T>
class model;

/**
 * @brief Base class for all neural network layers.
 *
 * This abstract class defines the common interface and behavior for all neural
 * network layers, including forward and backward passes, weight updates,...
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class Layer : public object {
public:
    explicit Layer(model<T>* m)
        : mModel(m)
    {
    }

    ~Layer() override = default;

    // Führt die Vorwärtspassage durch und schreibt das Ergebnis in output.
    virtual auto forwardPass(const T* input_data, T* output_data, bool trainingflag) -> ErrorType = 0;

    // Führt die Räckwärtspassage durch und berechnet die Gradienten für den vorherigen Layer.
    virtual auto backwardPass(const T* input_data, T* output_data) -> ErrorType = 0;

    virtual auto initGradients() -> ErrorType
    {
        return ErrorType::ok;
    }

    virtual auto deleteGradients() -> ErrorType
    {
        return ErrorType::ok;
    }

    // Update Weights and biases
    virtual auto update(uint32_t /* batchsize */) -> ErrorType
    {
        return ErrorType::ok;
    }

    virtual auto loadFromFlash() -> ErrorType = 0;

    virtual auto storeToFlash() -> ErrorType = 0;

    virtual auto getOutputSize() -> uint32_t = 0;

    virtual auto getInputSize() -> uint32_t = 0;

protected:
    // Flag that represents if the Layer has been loaded
    bool mIsLoaded = false;

    // pointer to the model Object that
    model<T>* mModel = nullptr;

    // pointer to Array with Input Data
    // Used for Layers like ReLU, softMax, Maxpool, ...
    // To have the Data for Backwardspass available
    T* mInputData;
};
} // namespace Edgeist

#endif // LAYER_H
