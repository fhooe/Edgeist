#pragma once
#include "Layer.h"
#include "Modeltypes.h"
#include "nmcf_ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include <vector>

/**
 * @author David Muttenthaler
 * @date 25-06-2025
 *
 * @brief Rectified Linear Unit (ReLU) Activation Layer.
 *
 * Applies the function `f(x) = max(0, x)` element-wise.
 * Adds non-linearity to the network and prevents vanishing gradients.
 */

template <typename T>
class model;

template <typename T>
class Relu :
	public Layer<T>
{
public:
	// Konstruktor: Initialisiert die ReLU-Schicht mit Zeigern auf Konfigurationsdaten und gewähltem Optimierer
	Relu(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType) : Layer<T>(m), mPtrLayer(HeaderPointer), mPtrData(DataPointer), mOptimizerType(OptimizerType) {


		this->mHeader = static_cast<Neural_Network_ReLU_t*>(mPtrLayer);
		this->mInputData = nullptr;

		loadFromFlash();

	}

	// Destruktor: Gibt dynamisch allokierten Speicher frei
	~Relu() {
		if (this->mInputData != nullptr) {
			delete[] this->mInputData;this->mInputData = nullptr;
		}
	}
	
	// Führt die Vorwärtspassage durch und schreibt das Ergebnis in output.
	virtual ErrorType forwardPass(const T* input_data, T* output_data, bool trainingflag) override {
		if (input_data == nullptr || output_data == nullptr)
		{
			return ErrorType::UnknownError;
		}


		if (trainingflag) {

			if (this->mIsLoaded != true) {
				return ErrorType::LayerNotInitialized;
			}
			// get memory for training
			if (this->mInputData == nullptr)
			{
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
	virtual ErrorType backwardPass(const T* input_data, T* output_data) override {

		if (input_data == nullptr || output_data == nullptr)
		{
			return ErrorType::UnknownError;
		}

		if (this->mIsLoaded != true) {
			return ErrorType::LayerNotInitialized;
		}

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

	// Lädt die trainierbaren werte vom Flash in den SRAM
	virtual ErrorType loadFromFlash() override {


		this->mIsLoaded = true;
		return ErrorType::ok;
	}

	// Speichert die trainierbaren werte vom SRAM in den Flash
	virtual ErrorType storeToFlash() override {
		this->mIsLoaded = false;
		// nicht implementier in dieser Version, wird erst am µController relevant
		return ErrorType::UnknownError;
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

private:
	void* mPtrLayer;

	void* mPtrData;

	// Pointer to the Layer in Flash
	Neural_Network_ReLU_t* mHeader;

	// chosen optimizer
	OptimizerID mOptimizerType;

};
