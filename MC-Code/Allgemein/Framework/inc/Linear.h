#pragma once
#include "Layer.h"
#include "Modeltypes.h"
#include "Modelstructs.h"
#include "ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include <vector>



template <typename T>
class Linear :
	public Layer<T>
{
public:
	//Tempor�r, bis typedef in "Modeltypes.h" integriert ist
	typedef T Linear_DataType_t;

	// Konstruktor: Initialisiert die Linear-Schicht mit Zeigern auf Konfigurationsdaten und gew�hltem Optimierer
	Linear(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType) : Layer<T>(m), mPtrLayer(HeaderPointer), mPtrData(DataPointer), mOptimizerType(OptimizerType) {

		mHeader = static_cast<Neural_Network_Linear_t*>(mPtrLayer);


		mPtrWeightFrozen = static_cast<Linear_DataType_t*>(mPtrLayer) + mHeader->weights_frozen_offset * sizeof(Linear_DataType_t);
		mPtrBiasFrozen = static_cast<Linear_DataType_t*>(mPtrLayer) + mHeader->bias_frozen_offset * sizeof(Linear_DataType_t);

		// Init Output values
		//loadFromFlash();

	}

	// Destruktor: Gibt dynamisch allokierten Speicher frei
	~Linear() {

		// Free memory

		if (mWeightPtr != nullptr) {
			delete[] mWeightPtr; mWeightPtr = nullptr;
		}

		if (mBiasPtr != nullptr) {
			delete[] mBiasPtr; mBiasPtr = nullptr;
		}

		if (this->mInputData != nullptr) {
			delete[] this->mInputData; this->mInputData = nullptr;
		}

		if (mPtrWeightGradient != nullptr) {
			delete[] mPtrWeightGradient; mPtrWeightGradient = nullptr;
		}

		if (mPtrBiasGradient != nullptr) {
			delete[] mPtrBiasGradient; mPtrBiasGradient = nullptr;
		}

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
			for (uint32_t i = 0; i < mHeader->dimensioninput_x; i++) {
				this->mInputData[i] = input_data[i];
			}
		}

		// Calc Result
		// step through outputs
		for (uint32_t b = 0; b < mHeader->dimensionoutput_x; b++) {

			Linear_DataType_t temp = Linear_DataType_t(0);
			for (uint32_t i = 0; i < mHeader->dimensioninput_x; i++) {
				temp += input_data[i] * mWeightPtr->getData((b*(mHeader->dimensioninput_x)) + i);
			}

			temp = mBiasPtr->getData(b) + temp;
			output_data[b] = temp;

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

		if (this->mInputData == nullptr) {
			return ErrorType::UnknownError;
		}


		//do Backward Pass
		T* dL_dW = new T[this->mHeader->dimensionoutput_x * this->mHeader->dimensioninput_x];
		for (int i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
			for (int j = 0; j < this->mHeader->dimensioninput_x; ++j) {
				dL_dW[i*this->mHeader->dimensioninput_x + j] = input_data[i] * this->mInputData[j];
			}
		}

		// 2. Gradient f�r Bias
		T* dL_db = new T[this->mHeader->dimensionoutput_x];
		for (int i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
			dL_db[i] = input_data[i];
		}

		// 3. Gradient f�r Input x (Backprop durch Layer)
		for (int j = 0; j < this->mHeader->dimensioninput_x; ++j) {
			output_data[j] = 0.0f;
			for (int i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
				output_data[j] += mWeightPtr->getData((i*(mHeader->dimensioninput_x)) + j) * input_data[i];
			}
		}

		//Update Gradients
		for (int i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
			for (int j = 0; j < this->mHeader->dimensioninput_x; ++j) {
				mPtrWeightGradient[i*this->mHeader->dimensioninput_x + j] += dL_dW[i*this->mHeader->dimensioninput_x + j];
			}
			mPtrBiasGradient[i] += dL_db[i];
		}

		if (this->mInputData != nullptr)
		{
			delete[] this->mInputData;
			this->mInputData = nullptr;
		}

		delete[] dL_dW; dL_dW = nullptr;
		delete[] dL_db; dL_db = nullptr;

		return ErrorType::ok;
	}

	// Initialisiere Gradienten, Speicher reservieren und initialisieren
	ErrorType initGradients() override {

		if (mPtrWeightGradient != nullptr || mPtrBiasGradient != nullptr)
		{
			return ErrorType::UnknownError;
		}

		// arrays dynamisch anlegen und mit 0.0 initialisieren

		uint32_t sizeWeights = this->mHeader->dimensionoutput_x * this->mHeader->dimensioninput_x;
		uint32_t sizeBias = this->mHeader->dimensionoutput_x;

		mPtrWeightGradient = new T[sizeWeights];
		mPtrBiasGradient = new T[sizeBias];

		for (uint32_t i = 0; i < sizeWeights; ++i) {
			mPtrWeightGradient[i] = T(0.0);
		}

		for (uint32_t i = 0; i < sizeBias; ++i) {
			mPtrBiasGradient[i] = T(0.0);
		}

		return ErrorType::ok;
	}

	// L�sche Gradienten, Speicher freigeben
	virtual ErrorType deleteGradients() override {
		if (mPtrWeightGradient != nullptr)
		{
			delete[] mPtrWeightGradient; mPtrWeightGradient = nullptr;
		}

		if (mPtrBiasGradient != nullptr)
		{
			delete[] mPtrBiasGradient; mPtrBiasGradient = nullptr;
		}

		return ErrorType::ok;
	}

	// Update Weights and biases
	virtual  ErrorType update(uint32_t batchsize) override{

		if (mPtrWeightGradient == nullptr || mPtrBiasGradient == nullptr)
		{
			return ErrorType::UnknownError;
		}

		// SGD-Update von W und b
		for (int i = 0; i < this->mHeader->dimensionoutput_x; ++i) {
			for (int j = 0; j < this->mHeader->dimensioninput_x; ++j) {
				mWeightPtr->update((i*(mHeader->dimensioninput_x)) + j, mPtrWeightGradient[i*this->mHeader->dimensioninput_x + j]/ batchsize, this->mModel->mLearningRate, mTimestep);
			}
			mBiasPtr->update(i, mPtrBiasGradient[i]/ batchsize, this->mModel->mLearningRate, mTimestep);
		}
		mTimestep++;
		return ErrorType::ok;
	}

	// L�dt die trainierbaren werte vom Flash in den SRAM
	virtual ErrorType loadFromFlash() override {

		switch (mOptimizerType) {
		case SGD:
			//initailisiere Weights
			mWeightPtr = new OptimizerSGD<T>;
			mWeightPtr->init(mHeader->weights_amount_trainable);
			//initailisiere Bias
			mBiasPtr = new OptimizerSGD<T>;
			mBiasPtr->init(mHeader->bias_amount_trainable);
			break;

		case Momentum:
			//initailisiere Weights
			mWeightPtr = new OptimizerMomentum<T>;
			mWeightPtr->init(mHeader->weights_amount_trainable);
			//initailisiere Bias
			mBiasPtr = new OptimizerMomentum<T>;
			mBiasPtr->init(mHeader->bias_amount_trainable);
			break;

		case ADAM:
			//initailisiere Weights
			mWeightPtr = new OptimizerAdam<T>;
			mWeightPtr->init(mHeader->weights_amount_trainable);
			//initailisiere Bias
			mBiasPtr = new OptimizerAdam<T>;
			mBiasPtr->init(mHeader->bias_amount_trainable);
			break;
		}

		//Initialize Weights und Biases 
		for (int i = 0; i < mHeader->weights_amount_trainable; i++) {
			mWeightPtr->setData(i, *(static_cast<Linear_DataType_t*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(Linear_DataType_t)) + i));
		}
		for (int i = 0; i < mHeader->bias_amount_trainable; i++) {
			mBiasPtr->setData(i, *(static_cast<Linear_DataType_t*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(Linear_DataType_t)) + i));
		}

		this->mIsLoaded = true;
		return ErrorType::ok;
	}

	// Speichert die trainierbaren werte vom SRAM in den Flash
	virtual ErrorType storeToFlash(Flash_manager* fm, uint32_t* start_address) override {

		// write weigths
		if (fm->WriteFlash((uint32_t)start_address + mHeader->weights_trainable_offset, (uint32_t*)mWeightPtr->getData(), mHeader->weights_amount_trainable) != Flash_Returntypes::Flash_OK)
		{
			return ErrorType::UnknownError;
		}
		
		// write bias
		if (fm->WriteFlash((uint32_t)start_address + mHeader->bias_trainable_offset, (uint32_t*)mBiasPtr->getData(), mHeader->bias_amount_trainable) != Flash_Returntypes::Flash_OK)
		{
			return ErrorType::UnknownError;
		}
		
		// delete Optimizer
		delete mWeightPtr;
		mWeightPtr = nullptr;
		delete mBiasPtr;
		mBiasPtr = nullptr;
		
		this->mIsLoaded = false;
		
		return ErrorType::ok;
	}

	virtual ErrorType storeExternalToFlash(Flash_manager* fm, uint32_t start_address_with_offset, uint32_t* data, uint32_t datasize, bool Weigth_or_Bias) override
	{
		uint32_t offset = Weigth_or_Bias ? mHeader->weights_trainable_offset : mHeader->bias_trainable_offset;
		if (fm->WriteFlash(start_address_with_offset + offset, data,datasize) != Flash_Returntypes::Flash_OK)
		{
			return ErrorType::UnknownError;
		}
		return ErrorType::ok;
	}

	virtual uint32_t getOutputSize() override {
		return uint32_t(mHeader->dimensionoutput_x);
	}

	virtual uint32_t getInputSize() override {
		return uint32_t(mHeader->dimensioninput_x);
	}

	const Neural_Network_Linear_t& header() {
		return *mHeader;
	}

	// get offestptr in Flash + amount
	virtual void getWeigths(uint32_t& Weigths_Offset, uint32_t& amount) override
	{
		if (mHeader != nullptr)
		{
			amount = mHeader->weights_amount_trainable;
			Weigths_Offset = mHeader->weights_trainable_offset;
		}
		else
		{
			amount = 0;
			Weigths_Offset = 0;
		}
	}

	// get offestptr in Flash + amount
	virtual void getBias(uint32_t& Bias_Offset, uint32_t& amount) override
	{
		if (mHeader != nullptr)
		{
			amount = mHeader->bias_amount_trainable;
			Bias_Offset = mHeader->bias_trainable_offset;
		}
		else
		{
			amount = 0;
			Bias_Offset = 0;
		}
	}

	virtual void deleteTrainableData(void) override
	{
		if (mWeightPtr != nullptr)
		{
			delete mWeightPtr;
			mWeightPtr = nullptr;
		}
		if (mBiasPtr != nullptr)
		{
			delete mBiasPtr;
			mBiasPtr = nullptr;
		}
		this->mIsLoaded = false;
	}

private:
	void* mPtrLayer;

	void* mPtrData;

	Neural_Network_Linear_t* mHeader;

	// chosen optimizer
	OptimizerID mOptimizerType;

	Linear_DataType_t* mPtrWeightFrozen;
	Linear_DataType_t* mPtrBiasFrozen;

	Linear_DataType_t* mPtrWeightGradient = nullptr;
	Linear_DataType_t* mPtrBiasGradient = nullptr;

	// vector with the trainabel Weigths in SRAM
	OptimizerBase<T>* mWeightPtr;

	OptimizerBase<T>* mBiasPtr;

	uint32_t mTimestep = 1;

};

