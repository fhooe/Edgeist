#pragma once
#include "object.h"
#include "ErrorTypes.h"
#include <stdint.h>
#include "flash_manager.h"


// Template-Forward-Deklaration statt include
//#include "model.h"
template <typename T>
class model;

// Basisklasse f�r alle Layer � abstrakte Schnittstelle
template <typename T>
class Layer {
public:

	explicit Layer(model<T>* m) : mModel(m) {}

	virtual ~Layer() = default;

	// F�hrt die Vorw�rtspassage durch und schreibt das Ergebnis in output.
	virtual ErrorType forwardPass(const T* input_data, T* output_data, bool train = false) = 0;

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

	virtual ErrorType storeToFlash(Flash_manager* fm, uint32_t* start_address) = 0;

	// Weigth_or_Bias:
	//	true = use Weigth_trainable_offset
	//	false = use Bias_trainable_offset
	virtual ErrorType storeExternalToFlash(Flash_manager* fm, uint32_t start_address_with_offset, uint32_t* data, uint32_t datasize, bool Weigth_or_Bias) = 0;

	virtual uint32_t getOutputSize() = 0;

	virtual uint32_t getInputSize() = 0;

	virtual void getWeigths(uint32_t& Weigths_Offset, uint32_t& amount) = 0;

	virtual void getBias(uint32_t& Bias_Offset, uint32_t& amount) = 0;

	virtual void deleteTrainableData(void) = 0;

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

