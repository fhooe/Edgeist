#pragma once
#include "Layer.h"
#include <iostream>
#include <iomanip>
#include "Modeltypes.h"
#include "ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include <vector>
#include <cmath>

template <typename T>
class model;

template <typename T>
class Softmax :
	public Layer<T>
{
public:
	// Konstruktor: Initialisiert die Softmax-Schicht mit Zeigern auf Konfigurationsdaten und gew�hltem Optimierer
	Softmax(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType) : Layer<T>(m), mPtrLayer(HeaderPointer), mPtrData(DataPointer), mOptimizerType(OptimizerType) {
		this->mHeader = static_cast<Neural_Network_Softmax_t*>(mPtrLayer);
		this->mInputData = nullptr;
	}

	// Destruktor: Gibt dynamisch allokierten Speicher frei
	~Softmax() {

		//Speicher freigeben
		delete[] this->mInputData;
		this->mInputData = nullptr;

	}

	// F�hrt die Vorw�rtspassage (Forward Pass) durch � berechnet Softmax-Ausgabe aus Eingabedaten
	// input_data: output der Vorherigen Layer
	// output_Data: output dieses Layers
	virtual ErrorType forwardPass(const T* input_data, T* output_data, bool train = false) override  {
		if (input_data == nullptr || output_data == nullptr)
		{
			return ErrorType::UnknownError;
		}

		if (this->mIsLoaded != true) {
			return ErrorType::LayerNotInitialized;
		}

		// Finde den Maximalwert der Eingabe zur numerischen Stabilisierung der Exponentialfunktion
		T maxVal = input_data[0];
		for (int i = 1; i < this->mHeader->dimensioninput_x; i++) {
			if (input_data[i] > maxVal) {
				maxVal = input_data[i];
			}
		}

		// Berechne exponentielle Werte der um maxVal verschobenen Eingaben und summiere sie
		T sumExponents = T(0);
		T* mPtrExponents = new T[this->mHeader->dimensioninput_x];

		for (int i = 0; i < this->mHeader->dimensioninput_x; i++) {
			mPtrExponents[i] = std::exp(input_data[i] - maxVal);
			sumExponents += mPtrExponents[i];
		}

		// Allokiere Speicher f�r Zwischenspeicherung der Eingabe (f�r Backpropagation)
		if (train)
		{
			if (this->mInputData == nullptr)
			{
				this->mInputData = new T[this->mHeader->dimensioninput_x];
			}
		}

		// Normiere Exponentialwerte zur Softmax-Ausgabe
		if (this->mInputData != nullptr) {
			for (int i = 0; i < this->mHeader->dimensionoutput_x; i++) {
				this->mInputData[i] = input_data[i];
			}
		}
		
		for (int i = 0; i < this->mHeader->dimensionoutput_x; i++) {
				// normieren
				output_data[i] = mPtrExponents[i] / sumExponents;
			}

		delete[] mPtrExponents;
		mPtrExponents = nullptr;

		//std::cout << "Ergebniss:" << std::endl;
		//std::cout << std::fixed << std::setprecision(10);

		//for (int i = 0; i < this->mHeader->dimensionoutput_x; i++) {
		//	std::cout << output_data[i] << std::endl;
		//}

		return ErrorType::ok;
	}

	// F�hrt die R�ckw�rtspassage (Backward Pass) durch � berechnet den Fehlergradienten der Softmax-Schicht
	// input_data: output vom forwardPass
	// output_Data: ist Gradient
	virtual ErrorType backwardPass(const T* input_data, T* output_data) override  {

		//// Berechne Loss (nur f�r Ausgabe; nicht Teil der Gradientenberechnung)
		//float loss = 0.0;
		//for (int i = 0; i < this->mHeader->dimensioninput_x; i++) {
		//	if (this->mModel->mPtrExpectedOutputData[i] > 0.0f) {
		//		loss -= std::log(input_data[i] + 1e-9f); // vermeidet log(0)
		//	}
		//}
		////+std::cout << std::fixed << std::setprecision(10);
		////std::cout << "Loss: " << loss << std::endl;

		if (input_data == nullptr || output_data == nullptr)
		{
			return ErrorType::UnknownError;
		}

		if (this->mIsLoaded != true) {
			return ErrorType::LayerNotInitialized;
		}

		// Softmax + Cross-Entropy Ableitung: p - y
		for (int i = 0; i < this->mHeader->dimensioninput_x; i++) {
			output_data[i] = input_data[i] - this->mModel->mPtrExpectedOutputData[i];
		}

		return ErrorType::ok;
	}

	// L�dt die trainierbaren werte vom Flash in den SRAM
	virtual ErrorType loadFromFlash() override  {

		this->mIsLoaded = true;
		// Nicht relevant bei Softmax, da keine Weights und Biases
		return ErrorType::ok;
	}

	// Speichert die trainierbaren werte vom SRAM in den Flash
	virtual ErrorType storeToFlash(Flash_manager* fm, uint32_t* start_address) override  {
		// Nicht relevant bei Softmax, da keine Weights und Biases
		return ErrorType::ok;
	}

	virtual ErrorType storeExternalToFlash(Flash_manager* fm, uint32_t start_address_with_offset, uint32_t* data, uint32_t datasize, bool Weigth_or_Bias) override
	{
		// Nicht relevant bei Softmax, da keine Weights und Biases
		return ErrorType::ok;
	}

	virtual uint32_t getOutputSize() override {
		return uint32_t(mHeader->dimensionoutput_x);
	}

	virtual uint32_t getInputSize() override {
		return uint32_t(mHeader->dimensioninput_x);
	}

	const Neural_Network_Softmax_t& header() {
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
	Neural_Network_Softmax_t* mHeader;

	// chosen optimizer
	OptimizerID mOptimizerType;

};