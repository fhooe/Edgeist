#pragma once
#include "object.h"
#include "nmcf_ErrorTypes.h"

/**
 * @author David Muttenthaler
 * @date 25-06-2025
 *
 * @brief Base class for all neural network layers.
 *
 * This abstract class defines the common interface and behavior for all neural
 * network layers, including forward and backward passes, weight updates,...
 */

// Template-Forward-Deklaration statt include
//#include "model.h"
template <typename T>
class model;

// Basisklasse f�r alle Layer � abstrakte Schnittstelle
template <typename T>
class Layer: public object {
public:

	explicit Layer(model<T>* m) : mModel(m) {}

	virtual ~Layer() = default;

	// F�hrt die Vorw�rtspassage durch und schreibt das Ergebnis in output.
	virtual ErrorType forwardPass(const T* input_data, T* output_data, bool trainingflag) = 0;

	// F�hrt die R�ckw�rtspassage durch und berechnet die Gradienten f�r den vorherigen Layer.
	virtual ErrorType backwardPass(const T* input_data, T* output_data) = 0;

	virtual ErrorType initGradients() {
		return ErrorType::ok;
	}

	virtual ErrorType deleteGradients() {
		return ErrorType::ok;
	}

	// Update Weights and biases
	virtual ErrorType update(uint32_t batchsize) {
		return ErrorType::ok;
	}

	virtual ErrorType loadFromFlash() = 0;

	virtual ErrorType storeToFlash() = 0;

	virtual uint32_t getOutputSize() = 0;

	virtual uint32_t getInputSize() = 0;


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

