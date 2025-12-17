#pragma once
#pragma once
#include "Layer.h"
#include "Modeltypes.h"
#include "Modelstructs.h"
#include "ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include <vector>

template <typename T>
class Conv2d :
	public Layer<T>
{
public:
	typedef T Conv2d_DataType_t;
	Conv2d(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType) : Layer<T>(m), mPtrLayer(HeaderPointer), mPtrData(DataPointer), mOptimizerType(OptimizerType) {

		mHeader = static_cast<Neural_Network_Conv2d_t*>(mPtrLayer);

		mPtrWeightFrozen = static_cast<Conv2d_DataType_t*>(mPtrLayer) + mHeader->weights_frozen_offset * sizeof(Conv2d_DataType_t);
		mPtrBiasFrozen = static_cast<Conv2d_DataType_t*>(mPtrLayer) + mHeader->bias_frozen_offset * sizeof(Conv2d_DataType_t);

		// load Weights and Biases
		//loadFromFlash();
	}

	~Conv2d() {
		// Free memory

		if (mWeightPtr != nullptr){
			delete[] mWeightPtr; mWeightPtr = nullptr;
		}

		if (mBiasPtr != nullptr){
			delete[] mBiasPtr; mBiasPtr = nullptr;
		}

		if (this->mInputData != nullptr){
			delete[] this->mInputData; this->mInputData = nullptr;
		}

		if (mPtrWeightGradient != nullptr){
			delete[] mPtrWeightGradient; mPtrWeightGradient = nullptr;
		}

		if (mPtrBiasGradient != nullptr){
			delete[] mPtrBiasGradient; mPtrBiasGradient = nullptr;
		}
	}

	//TODO
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
				this->mInputData = new T[mHeader->dimensioninput_x * mHeader->dimensioninput_y * mHeader->channelsin];
			}
			for (uint32_t i = 0; i < mHeader->dimensioninput_x * mHeader->dimensioninput_y * mHeader->channelsin; i++) {
				this->mInputData[i] = input_data[i];
			}
		}

		const int H = mHeader->dimensioninput_x;
		const int W = mHeader->dimensioninput_y;
		const int K[2] = { mHeader->kernelsize[0], mHeader->kernelsize[1] }; // beide dimensionen
		const int Cin = mHeader->channelsin;
		const int Cout = mHeader->channelsout;
		const int pad = K[0] / 2;  // "same" Padding

		// Zero-initialize output
		//std::fill(output_data, output_data + Cout * H * W, 0);

		// Schleife �ber alle Positionen im Output-Bild
		for (int oc = 0; oc < Cout; ++oc) {
			for (int oy = 0; oy < H; ++oy) {
				for (int ox = 0; ox < W; ++ox) {

					T sum = 0;

					// Schleife �ber alle Eingangskan�le und Kernelpositionen
					for (int ic = 0; ic < Cin; ++ic) {
						for (int ky = 0; ky < K[0]; ++ky) {
							for (int kx = 0; kx < K[1]; ++kx) {
								int iy = oy + ky - pad;
								int ix = ox + kx - pad;

								// Boundary check (zero padding)
								if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
									// Indices f�r Input, Gewicht, Output
									int input_idx = ((ic * H + iy) * W) + ix;
									int weight_idx = (((oc * Cin + ic) * K[0] + ky) * K[1] + kx);

									sum += input_data[input_idx] * mWeightPtr->getData(weight_idx);
								}
							}
						}
					}

					// Bias hinzuf�gen
					sum += mBiasPtr->getData(oc);

					// Output schreiben
					int output_idx = ((oc * H + oy) * W) + ox;
					output_data[output_idx] = sum;
				}
			}
		}
		return ErrorType::ok;
	}

	//TODO
	// F�hrt die R�ckw�rtspassage durch und berechnet die Gradienten f�r den vorherigen Layer.
	virtual ErrorType backwardPass(const T* input_data, T* output_data) override {

		const int H = mHeader->dimensioninput_x;
		const int W = mHeader->dimensioninput_y;
		const int K[2] = { mHeader->kernelsize[0], mHeader->kernelsize[1] };
		const int Cin = mHeader->channelsin;
		const int Cout = mHeader->channelsout;
		const int pad = K[0] / 2;  // same padding

		// Initialisiere Gradienten auf 0
		//for (int x = 0; x < K[0] * K[1] * Cin * Cout; x++) {
		//	mPtrWeightGradient[x] = T(0);
		//}

		//for (int x = 0; x < Cout; x++) {
		//	mPtrBiasGradient[x] = T(0);
		//}

		// dL/dW und dL/db berechnen
		for (int oc = 0; oc < Cout; ++oc) {
			for (int oy = 0; oy < H; ++oy) {
				for (int ox = 0; ox < W; ++ox) {
					int out_idx = ((oc * H + oy) * W + ox);
					T grad_out = input_data[out_idx];

					// Bias-Gradient: dL/db += dL/doutput
					mPtrBiasGradient[oc] += grad_out;

					for (int ic = 0; ic < Cin; ++ic) {
						for (int ky = 0; ky < K[0]; ++ky) {
							for (int kx = 0; kx < K[1]; ++kx) {
								int iy = oy + ky - pad;
								int ix = ox + kx - pad;

								if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
									int in_idx = ((ic * H + iy) * W + ix);
									int weight_idx = (((oc * Cin + ic) * K[0] + ky) * K[1] + kx);

									// dL/dW += dL/doutput * input
									mPtrWeightGradient[weight_idx] += this->mInputData[in_idx] * grad_out;
								}
							}
						}
					}
				}
			}
		}

		// dL/dinput berechnen (weiterreichen an vorherigen Layer)
		for (int ic = 0; ic < Cin; ++ic) {
			for (int iy = 0; iy < H; ++iy) {
				for (int ix = 0; ix < W; ++ix) {

					T sum = 0;

					for (int oc = 0; oc < Cout; ++oc) {
						for (int ky = 0; ky < K[0]; ++ky) {
							for (int kx = 0; kx < K[1]; ++kx) {
								int oy = iy - ky + pad;
								int ox = ix - kx + pad;

								if (oy >= 0 && oy < H && ox >= 0 && ox < W) {
									int out_idx = ((oc * H + oy) * W + ox);
									int weight_idx = (((oc * Cin + ic) * K[0] + ky) * K[1] + kx);

									// dL/dinput = SUM dL/doutput * W^T
									sum += input_data[out_idx] * mWeightPtr->getData(weight_idx);
								}
							}
						}
					}

					int in_grad_idx = ((ic * H + iy) * W + ix);
					output_data[in_grad_idx] = sum;
				}
			}
		}

		return ErrorType::ok;
	}

	// Initialisiere Gradienten, Speicher reservieren und initialisieren
	ErrorType initGradients() override {

		if (mPtrWeightGradient != nullptr || mPtrBiasGradient != nullptr)
		{
			return ErrorType::UnknownError;
		}

		// arrays dynamisch anlegen und mit 0.0 initialisieren

		uint32_t sizeWeights = mHeader->kernelsize[0] * mHeader->kernelsize[1] * mHeader->channelsin * mHeader->channelsout;
		uint32_t sizeBias = mHeader->channelsout;

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
	virtual  ErrorType update(uint32_t batchsize) override {

		if (mPtrWeightGradient == nullptr || mPtrBiasGradient == nullptr)
		{
			return ErrorType::UnknownError;
		}

		// Schleife �ber alle Filter (output channels)
		for (int oc = 0; oc < mHeader->channelsout; ++oc) {
			// Schleife �ber alle Eingangskan�le
			for (int ic = 0; ic < mHeader->channelsin; ++ic) {
				for (int ky = 0; ky < mHeader->kernelsize[0]; ++ky) {
					for (int kx = 0; kx < mHeader->kernelsize[1]; ++kx) {

						// Lineare Indexberechnung f�r 4D-Tensor flach gespeichert
						int index = (((oc * mHeader->channelsin + ic) * mHeader->kernelsize[0] + ky) * mHeader->kernelsize[1] + kx);

						mWeightPtr->update(index,
							mPtrWeightGradient[index] / batchsize,
							this->mModel->mLearningRate,
							mTimestep);
					}
				}
			}
			// Bias-Update: 1 Wert pro Filter
			mBiasPtr->update(oc,
				mPtrBiasGradient[oc] / batchsize,
				this->mModel->mLearningRate,
				mTimestep);
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
			mWeightPtr->setData(i, *(static_cast<Conv2d_DataType_t*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(Conv2d_DataType_t)) + i));
		}
		for (int i = 0; i < mHeader->bias_amount_trainable; i++) {
			mBiasPtr->setData(i, *(static_cast<Conv2d_DataType_t*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(Conv2d_DataType_t)) + i));
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
		return uint32_t(mHeader->dimensionoutput_x * mHeader->dimensionoutput_y * mHeader->channelsout);
	}

	virtual uint32_t getInputSize() override {
		return uint32_t(mHeader->dimensioninput_x * mHeader->dimensioninput_y * mHeader->channelsin);
	}

	const Neural_Network_Conv2d_t& header() {
		return *mHeader;
	}

	// get offset in Byte in Flash + amount
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

	Neural_Network_Conv2d_t* mHeader;

	// chosen optimizer
	OptimizerID mOptimizerType;

	Conv2d_DataType_t* mPtrWeightFrozen;
	Conv2d_DataType_t* mPtrBiasFrozen;

	Conv2d_DataType_t* mPtrWeightGradient = nullptr;
	Conv2d_DataType_t* mPtrBiasGradient = nullptr;

	// vector with the trainabel Weigths in SRAM
	OptimizerBase<T>* mWeightPtr;

	OptimizerBase<T>* mBiasPtr;

	uint32_t mTimestep = 1;

};

