#pragma once
#include "Layer.h"
#include "Modeltypes.h"
#include "ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include <vector>
#include <limits>

// Forward declaration of model
template <typename T>
class model;

/**
 * 2D Max-Pooling Layer implementation
 */
template <typename T>
class MaxPool2d : public Layer<T> {
public:
	MaxPool2d(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
		: Layer<T>(m), mPtrLayer(HeaderPointer), mPtrData(DataPointer), mOptimizerType(OptimizerType) {
		this->mHeader = static_cast<Neural_Network_MaxPool2d_t*>(mPtrLayer);
		mArgmax = nullptr;
	}

	~MaxPool2d() {
		delete[] mArgmax;
		mArgmax = nullptr;
	}

	virtual ErrorType forwardPass(const T* input_data, T* output_data, bool train = false) override {
		if (!input_data || !output_data) return ErrorType::UnknownError;
		if (!this->mIsLoaded) return ErrorType::LayerNotInitialized;

		const auto& H = *this->mHeader;
		const int C = H.channelsin;
		const int H_in = H.dimensioninput_y;
		const int W_in = H.dimensioninput_x;
		const int H_out = H.dimensionoutput_y;
		const int W_out = H.dimensionoutput_x;
		const int kH = H.kernelsize;
		const int kW = H.kernelsize;
		const int padH = H.padding;
		const int padW = H.padding;
		const int strideH = H.stride;
		const int strideW = H.stride;

		// allocate argmax storage
		size_t outSize = size_t(C) * H_out * W_out;
		delete[] mArgmax;
		mArgmax = new uint32_t[outSize];
		if (!mArgmax) return ErrorType::UnknownError;

		// iterate over channels and spatial dims
		for (int c = 0; c < C; ++c) {
			for (int oy = 0; oy < H_out; ++oy) {
				for (int ox = 0; ox < W_out; ++ox) {
					T maxVal = std::numeric_limits<T>::lowest();
					uint32_t maxIdx = 0;
					// window
					for (int ky = 0; ky < kH; ++ky) {
						for (int kx = 0; kx < kW; ++kx) {
							int inY = oy * strideH + ky - padH;
							int inX = ox * strideW + kx - padW;
							if (inY < 0 || inY >= H_in || inX < 0 || inX >= W_in) continue;
							size_t idx = size_t(c) * H_in * W_in + inY * W_in + inX;
							T val = input_data[idx];
							if (val > maxVal) {
								maxVal = val;
								maxIdx = idx;
							}
						}
					}
					size_t outIdx = size_t(c) * H_out * W_out + oy * W_out + ox;
					output_data[outIdx] = maxVal;
					mArgmax[outIdx] = maxIdx;
				}
			}
		}

		return ErrorType::ok;
	}

	virtual ErrorType backwardPass(const T* grad_output, T* grad_input) override {
		if (!grad_output || !grad_input) return ErrorType::UnknownError;
		if (!this->mIsLoaded) return ErrorType::LayerNotInitialized;
		if (!mArgmax) return ErrorType::UnknownError;

		const auto& H = *this->mHeader;
		const int C = H.channelsin;
		const int H_in = H.dimensioninput_y;
		const int W_in = H.dimensioninput_x;
		const int H_out = H.dimensionoutput_y;
		const int W_out = H.dimensionoutput_x;

		// zero initialize grad_input
		size_t inSize = size_t(C) * H_in * W_in;
		for (size_t i = 0; i < inSize; ++i) {
			grad_input[i] = T(0);
		}

		// propagate gradients
		for (size_t outIdx = 0; outIdx < size_t(C) * H_out * W_out; ++outIdx) {
			uint32_t inIdx = mArgmax[outIdx];
			grad_input[inIdx] += grad_output[outIdx];
		}

		return ErrorType::ok;
	}

	virtual ErrorType loadFromFlash() override {
		this->mIsLoaded = true;
		return ErrorType::ok;
	}

	virtual ErrorType storeToFlash(Flash_manager* fm,uint32_t* start_address) override {
		// not implemented for this version
		return ErrorType::ok;
	}

	virtual ErrorType storeExternalToFlash(Flash_manager* fm, uint32_t start_address_with_offset, uint32_t* data, uint32_t datasize, bool Weigth_or_Bias) override
	{
		return ErrorType::ok;
	}

	virtual uint32_t getOutputSize() override {
		return this->mHeader->channelsin * this->mHeader->dimensionoutput_x * this->mHeader->dimensionoutput_y;
	}

	virtual uint32_t getInputSize() override {
		return this->mHeader->channelsin * this->mHeader->dimensioninput_x * this->mHeader->dimensioninput_y;
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
	Neural_Network_MaxPool2d_t* mHeader;
	OptimizerID mOptimizerType;
	uint32_t* mArgmax;
};