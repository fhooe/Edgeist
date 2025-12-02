#pragma once
#include "Layer.h"
#include <iostream>
#include <iomanip>
#include "Modeltypes.h"
#include "nmcf_ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include <vector>
#include <cmath>

/**
 * @author David Muttenthaler
 * @date 25-06-2025
 *
 * @brief Softmax Activation Layer.
 *
 * Converts raw logits into probabilities by exponentiating and normalizing.
 * Used as the final layer in multi-class classification tasks.
 *
 * Output is a probability distribution (sums to 1).
 */

template <typename T>
class model;

template <typename T>
class Softmax :
	public Layer<T>
{
public:
	// Konstruktor: Initialisiert die Softmax-Schicht mit Zeigern auf Konfigurationsdaten und gewähltem Optimierer
	Softmax(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType) : Layer<T>(m), mPtrLayer(HeaderPointer), mPtrData(DataPointer), mOptimizerType(OptimizerType) {


		this->mHeader = static_cast<Neural_Network_Softmax_t*>(mPtrLayer);
		this->mInputData = nullptr;

		loadFromFlash();

	}

	// Destruktor: Gibt dynamisch allokierten Speicher frei
	~Softmax() {

		if (this->mInputData != nullptr) {
			delete[] this->mInputData;
			this->mInputData = nullptr;
		}
	}

	// Führt die Vorwärtspassage (Forward Pass) durch – berechnet Softmax-Ausgabe aus Eingabedaten
	// input_data: output der Vorherigen Layer
	// output_Data: output dieses Layers
	virtual ErrorType forwardPass(const T* input_data, T* output_data, bool trainingflag) override  {
		if (input_data == nullptr || output_data == nullptr)
		{
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

		//std::cout << "Ergebniss:" << std::endl;
		//std::cout << std::fixed << std::setprecision(10);

		//for (int i = 0; i < this->mHeader->dimensionoutput_x; i++) {
		//	std::cout << output_data[i] << std::endl;
		//}

		return ErrorType::ok;
	}

	// Führt die Rückwärtspassage (Backward Pass) durch – berechnet den Fehlergradienten der Softmax-Schicht
	// input_data: output vom forwardPass
	// output_Data: ist Gradient
	virtual ErrorType backwardPass(const T* input_data, T* output_data) override  {

		//// Berechne Loss (nur für Ausgabe; nicht Teil der Gradientenberechnung)
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
		for (size_t i = 0; i < this->mHeader->dimensioninput_x; i++) {
			output_data[i] = input_data[i] - this->mModel->mPtrExpectedOutputData[i];
		}

		return ErrorType::ok;
	}

	// Lädt die trainierbaren werte vom Flash in den SRAM
	virtual ErrorType loadFromFlash() override  {

		this->mIsLoaded = true;
		// Nicht relevant bei ReLU, da keine Weights und Biases
		return ErrorType::ok;
	}

	// Speichert die trainierbaren werte vom SRAM in den Flash
	virtual ErrorType storeToFlash() override  {
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

	const Neural_Network_Softmax_t& header() {
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
