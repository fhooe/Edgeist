/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the 2D convolutional layer.
 */

#ifndef CONV_2D_H
#define CONV_2D_H

#include "Modelstructs.h"
#include "layer.h"
#include "nmcf_error_types.h"
#include "optimizer_data_types.h"

namespace Edgeist {
/**
 * @brief 2D convolutional layer.
 *
 * Applies a 2D convolution over an input image or feature map using learnable
 * filters (kernels), bias terms, and stride/padding configuration.
 *
 * Used for feature extraction in image processing and computer vision tasks.
 * @tparam T Datatype of the layer inputs.
 */
template <typename T>
class Conv2D : public Layer<T> {
public:
    Conv2D(Model<T>* model, void* headerPointer, void* dataPointer, const OptimizerID optimizerType)
        : Layer<T>(model)
        , mPtrLayer(headerPointer)
        , mPtrData(dataPointer)
        , mHeader(static_cast<Neural_Network_Conv2d_t*>(mPtrLayer))
        , mOptimizerType(optimizerType)
    {
        mPtrWeightFrozen = static_cast<T*>(mPtrLayer) + mHeader->weights_frozen_offset * sizeof(T);
        mPtrBiasFrozen = static_cast<T*>(mPtrLayer) + mHeader->bias_frozen_offset * sizeof(T);

        mPtrFlashWeight = (static_cast<T*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(T)));
        mPtrFlashBias = (static_cast<T*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(T)));

        // Init Output values
        mLayerOutput = new T[mHeader->dimensionoutput_x * mHeader->dimensionoutput_y * mHeader->channelsout];

        Conv2D::loadFromFlash();
    }

    ~Conv2D() override
    {
        // Free memory
        if (mLayerOutput != nullptr) {
            delete[] mLayerOutput;
            mLayerOutput = nullptr;
        }

        if (mWeightPtr != nullptr) {
            delete[] mWeightPtr;
            mWeightPtr = nullptr;
        }

        if (mBiasPtr != nullptr) {
            delete[] mBiasPtr;
            mBiasPtr = nullptr;
        }

        if (this->mInputData != nullptr) {
            delete[] this->mInputData;
            this->mInputData = nullptr;
        }

        if (mPtrWeightGradient != nullptr) {
            delete[] mPtrWeightGradient;
            mPtrWeightGradient = nullptr;
        }

        if (mPtrBiasGradient != nullptr) {
            delete[] mPtrBiasGradient;
            mPtrBiasGradient = nullptr;
        }
    }

    // executes forward pass and writes the result to the output
    auto forwardPass(const T* inputData, T* outputData, const bool trainingFlag) -> ErrorType override
    {
        if (inputData == nullptr || outputData == nullptr) {
            return ErrorType::UnknownError;
        }

        // Use ether data from flash or SRAM based on trainingflag
        T* ptrWeight = nullptr;
        T* ptrBias = nullptr;
        if (trainingFlag) {
            // check if layer is loaded
            if (!this->mIsLoaded) {
                return ErrorType::LayerNotInitialized;
            }
            ptrWeight = mWeightPtr->mData;
            ptrBias = mBiasPtr->mData;
        } else {
            ptrWeight = mPtrFlashWeight;
            ptrBias = mPtrFlashBias;
        }

        // get memory for training
        if (trainingFlag) {
            this->mInputData = new T[mHeader->dimensioninput_x * mHeader->dimensioninput_y * mHeader->channelsin];

            for (uint32_t i = 0; i < mHeader->dimensioninput_x * mHeader->dimensioninput_y * mHeader->channelsin; i++) {
                this->mInputData[i] = inputData[i];
            }
        }

        const int H = mHeader->dimensioninput_x;
        const int W = mHeader->dimensioninput_y;
        const int K[2] = { mHeader->kernelsize[0], mHeader->kernelsize[1] }; // beide dimensionen
        const int Cin = mHeader->channelsin;
        const int Cout = mHeader->channelsout;
        const int pad = K[0] / 2; // "same" Padding

        // Zero-initialize output
        // std::fill(output_data, output_data + Cout * H * W, 0);

        // Loop through all positions in the output image
        for (int oc = 0; oc < Cout; ++oc) {
            for (int oy = 0; oy < H; ++oy) {
                for (int ox = 0; ox < W; ++ox) {

                    T sum = 0;

                    // Loop through all input channels and kernel positions
                    for (int ic = 0; ic < Cin; ++ic) {
                        for (int ky = 0; ky < K[0]; ++ky) {
                            for (int kx = 0; kx < K[1]; ++kx) {
                                int iy = oy + ky - pad;
                                int ix = ox + kx - pad;

                                // Boundary check (zero padding)
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                    // indices for input, weight, output
                                    int input_idx = ((ic * H + iy) * W) + ix;
                                    int weight_idx = (((oc * Cin + ic) * K[0] + ky) * K[1] + kx);

                                    sum += inputData[input_idx] * ptrWeight[weight_idx];
                                }
                            }
                        }
                    }

                    // add bias
                    sum += ptrBias[oc];

                    // write output
                    int output_idx = ((oc * H + oy) * W) + ox;
                    outputData[output_idx] = sum;
                }
            }
        }
        return ErrorType::ok;
    }

    // Performs the backwardpass and calculates the gradients for the previous layer
    auto backwardPass(const T* inputData, T* outputData) -> ErrorType override
    {
        const int H = mHeader->dimensioninput_x;
        const int W = mHeader->dimensioninput_y;
        const int K[2] = { mHeader->kernelsize[0], mHeader->kernelsize[1] };
        const int Cin = mHeader->channelsin;
        const int Cout = mHeader->channelsout;
        const int pad = K[0] / 2; // same padding

        // check if layer is loaded
        if (!this->mIsLoaded) {
            return ErrorType::LayerNotInitialized;
        }
        T* ptrWeight = mWeightPtr->mData;

        // calc dL/dW and dL/db
        for (int oc = 0; oc < Cout; ++oc) {
            for (int oy = 0; oy < H; ++oy) {
                for (int ox = 0; ox < W; ++ox) {
                    int out_idx = ((oc * H + oy) * W + ox);
                    T grad_out = inputData[out_idx];

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

        // calculate dL/dinput (pass to previous layer)
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
                                    sum += inputData[out_idx] * ptrWeight[weight_idx];
                                }
                            }
                        }
                    }

                    int in_grad_idx = ((ic * H + iy) * W + ix);
                    outputData[in_grad_idx] = sum;
                }
            }
        }

        // free memory
        delete[] this->mInputData;
        this->mInputData = nullptr;

        return ErrorType::ok;
    }

    // init gradient, reserve and init memory
    auto initGradients() -> ErrorType override
    {

        if (mPtrWeightGradient != nullptr || mPtrBiasGradient != nullptr) {
            return ErrorType::UnknownError;
        }

        // create arrays dynamically and initialize them with 0.0
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

    // delete gradient and free memory
    auto deleteGradients() -> ErrorType override
    {
        if (mPtrWeightGradient != nullptr) {
            delete[] mPtrWeightGradient;
            mPtrWeightGradient = nullptr;
        }

        if (mPtrBiasGradient != nullptr) {
            delete[] mPtrBiasGradient;
            mPtrBiasGradient = nullptr;
        }

        return ErrorType::ok;
    }

    // update weights and biases
    auto update(uint32_t batchsize) -> ErrorType override
    {

        if (mPtrWeightGradient == nullptr || mPtrBiasGradient == nullptr) {
            return ErrorType::UnknownError;
        }

        // Loop through all filters (output channels)
        for (int oc = 0; oc < mHeader->channelsout; ++oc) {
            // Loop through all input channels
            for (int ic = 0; ic < mHeader->channelsin; ++ic) {
                for (int ky = 0; ky < mHeader->kernelsize[0]; ++ky) {
                    for (int kx = 0; kx < mHeader->kernelsize[1]; ++kx) {

                        // Linear index calculation for 4D tensor stored flat
                        int index = (((oc * mHeader->channelsin + ic) * mHeader->kernelsize[0] + ky) * mHeader->kernelsize[1] + kx);

                        mWeightPtr->update(index,
                            mPtrWeightGradient[index] / batchsize,
                            this->mModel->mLearningRate,
                            mTimestep);
                    }
                }
            }
            // Bias-Update: 1 value per filter
            mBiasPtr->update(oc,
                mPtrBiasGradient[oc] / batchsize,
                this->mModel->mLearningRate,
                mTimestep);
        }
        mTimestep++;
        return ErrorType::ok;
    }

    // Loads the trainable values from Flash into SRAM
    auto loadFromFlash() -> ErrorType override
    {
        switch (mOptimizerType) {
        case OptimizerID::SGD:
            // init weights
            mWeightPtr = new OptimizerSGD<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // init bias
            mBiasPtr = new OptimizerSGD<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;

        case OptimizerID::Momentum:
            // init weights
            mWeightPtr = new OptimizerMomentum<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // init bias
            mBiasPtr = new OptimizerMomentum<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;

        case OptimizerID::ADAM:
            // init weights
            mWeightPtr = new OptimizerAdam<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // init bias
            mBiasPtr = new OptimizerAdam<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;
        }

        // init weights and biases
        for (size_t i = 0; i < mHeader->weights_amount_trainable; i++) {
            mWeightPtr->setData(i, *(static_cast<T*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(T)) + i));
        }
        for (size_t i = 0; i < mHeader->bias_amount_trainable; i++) {
            mBiasPtr->setData(i, *(static_cast<T*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(T)) + i));
        }

        this->mIsLoaded = true;
        return ErrorType::ok;
    }

    // Stores the trainable values from SRAM to Flash
    auto storeToFlash() -> ErrorType override
    {
        this->mIsLoaded = false;
        // Not implemented in this version, will only become relevant on the uController
        return ErrorType::UnknownError;
    }

    auto getOutputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensionoutput_x * mHeader->dimensionoutput_y * mHeader->channelsout);
    }

    auto getInputSize() -> uint32_t override
    {
        return uint32_t(mHeader->dimensioninput_x * mHeader->dimensioninput_y * mHeader->channelsin);
    }

    [[nodiscard]] auto header() const -> const Neural_Network_Conv2d_t&
    {
        return *mHeader;
    }

private:
    void* mPtrLayer;

    void* mPtrData;

    Neural_Network_Conv2d_t* mHeader;

    // chosen optimizer
    OptimizerID mOptimizerType;

    T* mPtrWeightFrozen;
    T* mPtrBiasFrozen;

    T* mPtrFlashWeight;
    T* mPtrFlashBias;

    T* mPtrWeightGradient = nullptr;
    T* mPtrBiasGradient = nullptr;

    // vector with output Values
    T* mLayerOutput;

    // vector with the trainable weights in SRAM
    OptimizerBase<T>* mWeightPtr;

    OptimizerBase<T>* mBiasPtr;

    uint32_t mTimestep = 1;
};
} // namespace Edgeist

#endif // CONV_2D_H
