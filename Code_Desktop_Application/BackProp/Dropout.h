#pragma once
#include "Layer.h"
#include "Modeltypes.h"
#include "nmcf_ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include <random>
#include <vector>

/**
 * @author David Muttenthaler
 * @date 25-06-2025
 *
 * @brief Dropout Layer.
 *
 * Randomly zeroes some of the elements of the input tensor during training
 * with a probability `p`, helping prevent overfitting.
 *
 * No effect during inference (pass-through).
 */
template <typename T>
class Dropout : public Layer<T> {
public:
	Dropout(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
		: Layer<T>(m), mPtrLayer(HeaderPointer), mPtrData(DataPointer), mOptimizerType(OptimizerType),
		mRng(std::random_device{}()) {
		this->mHeader = static_cast<Neural_Network_Dropout_t*>(mPtrLayer);
		loadFromFlash();
	}

	virtual ErrorType forwardPass(const T* input_data, T* output_data, bool trainingflag) override {
		if (!input_data || !output_data) return ErrorType::UnknownError;
		if (!this->mIsLoaded) return ErrorType::LayerNotInitialized;

		const size_t size = mHeader->dimensioninput_x;

		if (trainingflag) {
			if (mDropoutMask.size() != size) return ErrorType::DropoutMaskMissing;

			for (size_t i = 0; i < size; ++i) {
				output_data[i] = input_data[i] * mDropoutMask[i];
			}
		}
		else {
			for (size_t i = 0; i < size; ++i) {
				output_data[i] = input_data[i]; // Keine Änderung im Inferenzmodus
			}
		}

		return ErrorType::ok;
	}

	virtual ErrorType backwardPass(const T* grad_output, T* grad_input) override {
		if (!grad_output || !grad_input) return ErrorType::UnknownError;
		if (!this->mIsLoaded) return ErrorType::LayerNotInitialized;

		const size_t size = mHeader->dimensioninput_x;

		if (mDropoutMask.size() != size) return ErrorType::DropoutMaskMissing;

		for (size_t i = 0; i < size; ++i) {
			grad_input[i] = grad_output[i] * mDropoutMask[i];
		}

		return ErrorType::ok;
	}

	virtual ErrorType initGradients() override {
		const auto& H = *this->mHeader;
		const size_t size = size_t(H.dimensioninput_x);
		const float rate = H.dropoutrate;

		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		mDropoutMask.resize(size);

		for (size_t i = 0; i < size; ++i) {
			bool keep = dist(mRng) >= rate;
			mDropoutMask[i] = keep ? T(1.0f) / (1.0f - rate) : T(0);
		}

		return ErrorType::ok;
	}

	virtual ErrorType deleteGradients() override {
		mDropoutMask.clear();
		return ErrorType::ok;
	}

	virtual ErrorType loadFromFlash() override {
		this->mIsLoaded = true;
		return ErrorType::ok;
	}

	virtual ErrorType storeToFlash() override {
		// Not implemented
		return ErrorType::ok;
	}

	virtual uint32_t getOutputSize() override {
		return mHeader->dimensionoutput_x;
	}

	virtual uint32_t getInputSize() override {
		return mHeader->dimensioninput_x;
	}

private:
	void* mPtrLayer;
	void* mPtrData;
	Neural_Network_Dropout_t* mHeader;
	OptimizerID mOptimizerType;

	std::vector<T> mDropoutMask;
	std::mt19937 mRng;
};
