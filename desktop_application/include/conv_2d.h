/**
 * @author David Muttenthaler
 * @brief Implements the 2D convolutional layer.
 **/

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
class Conv2d : public Layer<T> {
public:
    typedef T Conv2d_DataType_t;
    Conv2d(model<T>* m, void* HeaderPointer, void* DataPointer, OptimizerID OptimizerType)
        : Layer<T>(m)
        , mPtrLayer(HeaderPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {

        mHeader = static_cast<Neural_Network_Conv2d_t*>(mPtrLayer);

        mPtrWeightFrozen = static_cast<Conv2d_DataType_t*>(mPtrLayer) + mHeader->weights_frozen_offset * sizeof(Conv2d_DataType_t);
        mPtrBiasFrozen = static_cast<Conv2d_DataType_t*>(mPtrLayer) + mHeader->bias_frozen_offset * sizeof(Conv2d_DataType_t);

        mPtrFlashWeight = (static_cast<Conv2d_DataType_t*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(Conv2d_DataType_t)));
        mPtrFlashBias = (static_cast<Conv2d_DataType_t*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(Conv2d_DataType_t)));

        // Init Output values
        mLayerOutput = new Conv2d_DataType_t[mHeader->dimensionoutput_x * mHeader->dimensionoutput_y * mHeader->channelsout];

        loadFromFlash();
    }

    ~Conv2d() override
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

    // Führt die Vorwärtspassage durch und schreibt das Ergebnis in output.
    ErrorType forwardPass(const T* input_data, T* output_data, bool trainingflag) override
    {
        if (input_data == nullptr || output_data == nullptr) {
            return ErrorType::UnknownError;
        }

        // Use ether data from flash or SRAM based on trainingflag
        T* ptrWeight = nullptr;
        T* ptrBias = nullptr;
        if (trainingflag) {
            // check if layer is loaded
            if (this->mIsLoaded != true) {
                return ErrorType::LayerNotInitialized;
            }
            ptrWeight = mWeightPtr->mData;
            ptrBias = mBiasPtr->mData;
        } else {
            ptrWeight = mPtrFlashWeight;
            ptrBias = mPtrFlashBias;
        }

        // get memory for training
        if (trainingflag) {
            this->mInputData = new T[mHeader->dimensioninput_x * mHeader->dimensioninput_y * mHeader->channelsin];

            for (uint32_t i = 0; i < mHeader->dimensioninput_x * mHeader->dimensioninput_y * mHeader->channelsin; i++) {
                this->mInputData[i] = input_data[i];
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

        // Schleife über alle Positionen im Output-Bild
        for (int oc = 0; oc < Cout; ++oc) {
            for (int oy = 0; oy < H; ++oy) {
                for (int ox = 0; ox < W; ++ox) {

                    T sum = 0;

                    // Schleife über alle Eingangskanäle und Kernelpositionen
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

                                    sum += input_data[input_idx] * ptrWeight[weight_idx];
                                }
                            }
                        }
                    }

                    // Bias hinzuf�gen
                    sum += ptrBias[oc];

                    // Output schreiben
                    int output_idx = ((oc * H + oy) * W) + ox;
                    output_data[output_idx] = sum;
                }
            }
        }
        return ErrorType::ok;
    }

    // Führt die Rückwärtspassage durch und berechnet die Gradienten für den vorherigen Layer.
    ErrorType backwardPass(const T* input_data, T* output_data) override
    {

        const int H = mHeader->dimensioninput_x;
        const int W = mHeader->dimensioninput_y;
        const int K[2] = { mHeader->kernelsize[0], mHeader->kernelsize[1] };
        const int Cin = mHeader->channelsin;
        const int Cout = mHeader->channelsout;
        const int pad = K[0] / 2; // same padding

        // check if layer is loaded
        if (this->mIsLoaded != true) {
            return ErrorType::LayerNotInitialized;
        }
        T* ptrWeight = mWeightPtr->mData;

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
                                    sum += input_data[out_idx] * ptrWeight[weight_idx];
                                }
                            }
                        }
                    }

                    int in_grad_idx = ((ic * H + iy) * W + ix);
                    output_data[in_grad_idx] = sum;
                }
            }
        }

        // free memory
        delete[] this->mInputData;
        this->mInputData = nullptr;

        return ErrorType::ok;
    }

    // Initialisiere Gradienten, Speicher reservieren und initialisieren
    ErrorType initGradients() override
    {

        if (mPtrWeightGradient != nullptr || mPtrBiasGradient != nullptr) {
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

    // Lösche Gradienten, Speicher freigeben
    ErrorType deleteGradients() override
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

    // Update Weights and biases
    ErrorType update(uint32_t batchsize) override
    {

        if (mPtrWeightGradient == nullptr || mPtrBiasGradient == nullptr) {
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

    // Lädt die trainierbaren werte vom Flash in den SRAM
    ErrorType loadFromFlash() override
    {

        switch (mOptimizerType) {
        case SGD:
            // initailisiere Weights
            mWeightPtr = new OptimizerSGD<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // initailisiere Bias
            mBiasPtr = new OptimizerSGD<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;

        case Momentum:
            // initailisiere Weights
            mWeightPtr = new OptimizerMomentum<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // initailisiere Bias
            mBiasPtr = new OptimizerMomentum<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;

        case ADAM:
            // initailisiere Weights
            mWeightPtr = new OptimizerAdam<T>;
            mWeightPtr->init(mHeader->weights_amount_trainable);
            // initailisiere Bias
            mBiasPtr = new OptimizerAdam<T>;
            mBiasPtr->init(mHeader->bias_amount_trainable);
            break;
        }

        // Initialize Weights und Biases
        for (size_t i = 0; i < mHeader->weights_amount_trainable; i++) {
            mWeightPtr->setData(i, *(static_cast<Conv2d_DataType_t*>(mPtrData) + (mHeader->weights_trainable_offset / sizeof(Conv2d_DataType_t)) + i));
        }
        for (size_t i = 0; i < mHeader->bias_amount_trainable; i++) {
            mBiasPtr->setData(i, *(static_cast<Conv2d_DataType_t*>(mPtrData) + (mHeader->bias_trainable_offset / sizeof(Conv2d_DataType_t)) + i));
        }

        this->mIsLoaded = true;
        return ErrorType::ok;
    }

    // Speichert die trainierbaren werte vom SRAM in den Flash
    ErrorType storeToFlash() override
    {
        this->mIsLoaded = false;
        // nicht implementier in dieser Version, wird erst am �Controller relevant
        return ErrorType::UnknownError;
    }

    uint32_t getOutputSize() override
    {
        return uint32_t(mHeader->dimensionoutput_x * mHeader->dimensionoutput_y * mHeader->channelsout);
    }

    uint32_t getInputSize() override
    {
        return uint32_t(mHeader->dimensioninput_x * mHeader->dimensioninput_y * mHeader->channelsin);
    }

    const Neural_Network_Conv2d_t& header()
    {
        return *mHeader;
    }

private:
    void* mPtrLayer;

    void* mPtrData;

    Neural_Network_Conv2d_t* mHeader;

    // chosen optimizer
    OptimizerID mOptimizerType;

    Conv2d_DataType_t* mPtrWeightFrozen;
    Conv2d_DataType_t* mPtrBiasFrozen;

    Conv2d_DataType_t* mPtrFlashWeight;
    Conv2d_DataType_t* mPtrFlashBias;

    Conv2d_DataType_t* mPtrWeightGradient = nullptr;
    Conv2d_DataType_t* mPtrBiasGradient = nullptr;

    // vector with output Values
    Conv2d_DataType_t* mLayerOutput;

    // vector with the trainabel Weigths in SRAM
    OptimizerBase<T>* mWeightPtr;

    OptimizerBase<T>* mBiasPtr;

    uint32_t mTimestep = 1;
};
} // namespace Edgeist

#endif // CONV_2D_H
