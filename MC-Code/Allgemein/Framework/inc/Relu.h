#pragma once
#include "Layer.h"
#include "Modeltypes.h"
#include "ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include <vector>

template <typename T>
class model;

template <typename T>
class Relu :
	public Layer<T>
{
public:
	// Konstruktor: Initialisiert die ReLU-Schicht mit Zeigern auf Konfigurationsdaten und gew�hltem Optimierer
	Relu(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType) : Layer<T>(m), mPtrLayer(HeaderPointer), mPtrData(DataPointer), mOptimizerType(OptimizerType) {
		this->mHeader = static_cast<Neural_Network_ReLU_t*>(mPtrLayer);
		this->mInputData = nullptr;
	}

	// Destruktor: Gibt dynamisch allokierten Speicher frei
	~Relu() {
		delete[] this->mInputData;
		this->mInputData = nullptr;
	}
	
	// F�hrt die Vorw�rtspassage durch und schreibt das Ergebnis in output.
	virtual ErrorType forwardPass(const T* input_data, T* output_data, bool train = false) override {
		if (input_data == nullptr || output_data == nullptr)
		{
			return ErrorType::UnknownError;
		}

		if (this->mIsLoaded != true) {
			return ErrorType::LayerNotInitialized;
		}

		// get memory for training
		if (train)
		{
			if (this->mInputData == nullptr)
			{
				this->mInputData = new T[this->mHeader->dimensioninput_x];
			}
		}
		
		if (this->mInputData != nullptr) {
			for (int i = 0; i < this->mHeader->dimensionoutput_x; i++) {
				this->mInputData[i] = input_data[i];
			}
		}

		for (int i = 0; i < this->mHeader->dimensionoutput_x; i++) {
				output_data[i] = (input_data[i] > T(0)) ? input_data[i] : T(0);
		}
		return ErrorType::ok;
	}

	// F�hrt die R�ckw�rtspassage durch und berechnet die Gradienten f�r den vorherigen Layer.
	virtual ErrorType backwardPass(const T* input_data, T* output_data) override {

		if (input_data == nullptr || output_data == nullptr)
		{
			return ErrorType::UnknownError;
		}

		if (this->mIsLoaded != true) {
			return ErrorType::LayerNotInitialized;
		}

		T* calc_input;
		if (this->mInputData == nullptr)
		{
			return ErrorType::UnknownError;
		}


		for (uint32_t i = 0; i < this->mHeader->dimensioninput_x; i++)
		{
			output_data[i] = (this->mInputData[i] > T(0)) ? input_data[i] : T(0);
		}

		if (this->mInputData != nullptr)
		{
			delete[] this->mInputData;
			this->mInputData = nullptr;
		}

		return ErrorType::UnknownError;
	}

	// L�dt die trainierbaren werte vom Flash in den SRAM
	virtual ErrorType loadFromFlash() override {
		this->mIsLoaded = true;
		// Nicht relevant bei ReLU, da keine Weights und Biases
		return ErrorType::ok;
	}

	// Speichert die trainierbaren werte vom SRAM in den Flash
	virtual ErrorType storeToFlash(Flash_manager* fm, uint32_t* start_address) override {
		// Nicht relevant bei ReLU, da keine Weights und Biases
		return ErrorType::ok;
	}

	virtual ErrorType storeExternalToFlash(Flash_manager* fm, uint32_t start_address_with_offset, uint32_t* data, uint32_t datasize, bool Weigth_or_Bias) override
	{
		// Nicht relevant bei ReLU, da keine Weights und Biases
		return ErrorType::ok;
	}

	virtual uint32_t getOutputSize() override {
		return uint32_t(mHeader->dimensionoutput_x);
	}

	virtual uint32_t getInputSize() override {
		return uint32_t(mHeader->dimensioninput_x);
	}

	const Neural_Network_ReLU_t& header() {
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
	Neural_Network_ReLU_t* mHeader;

	// chosen optimizer
	OptimizerID mOptimizerType;

};