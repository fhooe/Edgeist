#pragma once
#include "Layer.h"
#include "Modeltypes.h"
#include "ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include <vector>

template <typename T>
class model;

template <typename T>
class Flatten :
	public Layer<T>
{
public:
	// Konstruktor: Initialisiert die ReLU-Schicht mit Zeigern auf Konfigurationsdaten und gew�hltem Optimierer
	Flatten(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType) : Layer<T>(m), mPtrLayer(HeaderPointer), mPtrData(DataPointer), mOptimizerType(OptimizerType) {

		this->mHeader = static_cast<Neural_Network_Flatten_t*>(mPtrLayer);
	}

	// Destruktor: Gibt dynamisch allokierten Speicher frei
	~Flatten() {

	}
	
	// F�hrt die Vorw�rtspassage durch und schreibt das Ergebnis in output.
	virtual ErrorType forwardPass(const T* input_data, T* output_data, bool train = false) override {
		for (int i = 0; i < mHeader->dimensioninput_x; i++) {
			output_data[i] = input_data[i];
		}
		return ErrorType::ok;
	}

	// F�hrt die R�ckw�rtspassage durch und berechnet die Gradienten f�r den vorherigen Layer.
	virtual ErrorType backwardPass(const T* input_data, T* output_data) override {
		for (int i = 0; i < mHeader->dimensioninput_x; i++) {
			output_data[i] = input_data[i];
		}
		return ErrorType::ok;
	}

	// L�dt die trainierbaren werte vom Flash in den SRAM
	virtual ErrorType loadFromFlash() override {;
		return ErrorType::ok;
	}

	// Speichert die trainierbaren werte vom SRAM in den Flash
	virtual ErrorType storeToFlash(Flash_manager* fm, uint32_t* start_address) override {
		return ErrorType::ok;
	}

	virtual ErrorType storeExternalToFlash(Flash_manager* fm, uint32_t start_address_with_offset, uint32_t* data, uint32_t datasize, bool Weigth_or_Bias) override
	{
		return ErrorType::ok;
	}

	virtual uint32_t getOutputSize() override {
		return uint32_t(mHeader->dimensionoutput_x);
	}

	virtual uint32_t getInputSize() override {
		return uint32_t(mHeader->dimensioninput_x);
	}

	const Neural_Network_Flatten_t& header() {
		return *mHeader;
	}

	virtual void getWeigths(uint32_t& Weigths_Offset, uint32_t& amount) override
	{
		amount = 0;
		Weigths_Offset = 0;
	}

	virtual void getBias(uint32_t& Bias_Offset, uint32_t& amount) override
	{
		amount = 0;
		Bias_Offset = 0;
	}

	virtual void deleteTrainableData(void) override
	{
		// no trainable data to delete
	}

private:
	void* mPtrLayer;

	void* mPtrData;

	// Pointer to the Layer in Flash
	Neural_Network_Flatten_t* mHeader;

	// chosen optimizer
	OptimizerID mOptimizerType;

};