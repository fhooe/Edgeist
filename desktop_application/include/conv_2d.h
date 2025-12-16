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
        , m_ptrLayer(headerPointer)
        , m_ptrData(dataPointer)
        , m_header(static_cast<Neural_Network_Conv2d_t*>(m_ptrLayer))
        , m_optimizerType(optimizerType)
    {
        m_ptrWeightFrozen = static_cast<T*>(m_ptrLayer) + m_header->weights_frozen_offset * sizeof(T);
        m_ptrBiasFrozen = static_cast<T*>(m_ptrLayer) + m_header->bias_frozen_offset * sizeof(T);

        m_ptrFlashWeight = (static_cast<T*>(m_ptrData) + (m_header->weights_trainable_offset / sizeof(T)));
        m_ptrFlashBias = (static_cast<T*>(m_ptrData) + (m_header->bias_trainable_offset / sizeof(T)));

        // Init Output values
        m_layerOutput = new T[m_header->dimensionoutput_x * m_header->dimensionoutput_y * m_header->channelsout];

        Conv2D::loadFromFlash();
    }

    ~Conv2D() override
    {
        // Free memory
        if (m_layerOutput != nullptr) {
            delete[] m_layerOutput;
            m_layerOutput = nullptr;
        }

        if (m_ptrWeight != nullptr) {
            delete[] m_ptrWeight;
            m_ptrWeight = nullptr;
        }

        if (m_ptrBias != nullptr) {
            delete[] m_ptrBias;
            m_ptrBias = nullptr;
        }

        if (this->m_inputData != nullptr) {
            delete[] this->m_inputData;
            this->m_inputData = nullptr;
        }

        if (m_ptrWeightGradient != nullptr) {
            delete[] m_ptrWeightGradient;
            m_ptrWeightGradient = nullptr;
        }

        if (m_ptrBiasGradient != nullptr) {
            delete[] m_ptrBiasGradient;
            m_ptrBiasGradient = nullptr;
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
            if (!this->m_isLoaded) {
                return ErrorType::LayerNotInitialized;
            }
            ptrWeight = m_ptrWeight->m_data;
            ptrBias = m_ptrBias->m_data;
        } else {
            ptrWeight = m_ptrFlashWeight;
            ptrBias = m_ptrFlashBias;
        }

        // get memory for training
        if (trainingFlag) {
            this->m_inputData = new T[m_header->dimensioninput_x * m_header->dimensioninput_y * m_header->channelsin];

            for (uint32_t i = 0; i < m_header->dimensioninput_x * m_header->dimensioninput_y * m_header->channelsin; i++) {
                this->m_inputData[i] = inputData[i];
            }
        }

        const int H = m_header->dimensioninput_x;
        const int W = m_header->dimensioninput_y;
        const int K[2] = { m_header->kernelsize[0], m_header->kernelsize[1] }; // beide dimensionen
        const int Cin = m_header->channelsin;
        const int Cout = m_header->channelsout;
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
        return ErrorType::OK;
    }

    // Performs the backwardpass and calculates the gradients for the previous layer
    auto backwardPass(const T* inputData, T* outputData) -> ErrorType override
    {
        const int H = m_header->dimensioninput_x;
        const int W = m_header->dimensioninput_y;
        const int K[2] = { m_header->kernelsize[0], m_header->kernelsize[1] };
        const int Cin = m_header->channelsin;
        const int Cout = m_header->channelsout;
        const int pad = K[0] / 2; // same padding

        // check if layer is loaded
        if (!this->m_isLoaded) {
            return ErrorType::LayerNotInitialized;
        }
        T* ptrWeight = m_ptrWeight->m_data;

        // calc dL/dW and dL/db
        for (int oc = 0; oc < Cout; ++oc) {
            for (int oy = 0; oy < H; ++oy) {
                for (int ox = 0; ox < W; ++ox) {
                    int out_idx = ((oc * H + oy) * W + ox);
                    T grad_out = inputData[out_idx];

                    // Bias-Gradient: dL/db += dL/doutput
                    m_ptrBiasGradient[oc] += grad_out;

                    for (int ic = 0; ic < Cin; ++ic) {
                        for (int ky = 0; ky < K[0]; ++ky) {
                            for (int kx = 0; kx < K[1]; ++kx) {
                                int iy = oy + ky - pad;
                                int ix = ox + kx - pad;

                                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                    int in_idx = ((ic * H + iy) * W + ix);
                                    int weight_idx = (((oc * Cin + ic) * K[0] + ky) * K[1] + kx);

                                    // dL/dW += dL/doutput * input
                                    m_ptrWeightGradient[weight_idx] += this->m_inputData[in_idx] * grad_out;
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
        delete[] this->m_inputData;
        this->m_inputData = nullptr;

        return ErrorType::OK;
    }

    // init gradient, reserve and init memory
    auto initGradients() -> ErrorType override
    {

        if (m_ptrWeightGradient != nullptr || m_ptrBiasGradient != nullptr) {
            return ErrorType::UnknownError;
        }

        // create arrays dynamically and initialize them with 0.0
        uint32_t sizeWeights = m_header->kernelsize[0] * m_header->kernelsize[1] * m_header->channelsin * m_header->channelsout;
        uint32_t sizeBias = m_header->channelsout;

        m_ptrWeightGradient = new T[sizeWeights];
        m_ptrBiasGradient = new T[sizeBias];

        for (uint32_t i = 0; i < sizeWeights; ++i) {
            m_ptrWeightGradient[i] = T(0.0);
        }

        for (uint32_t i = 0; i < sizeBias; ++i) {
            m_ptrBiasGradient[i] = T(0.0);
        }

        return ErrorType::OK;
    }

    // delete gradient and free memory
    auto deleteGradients() -> ErrorType override
    {
        if (m_ptrWeightGradient != nullptr) {
            delete[] m_ptrWeightGradient;
            m_ptrWeightGradient = nullptr;
        }

        if (m_ptrBiasGradient != nullptr) {
            delete[] m_ptrBiasGradient;
            m_ptrBiasGradient = nullptr;
        }

        return ErrorType::OK;
    }

    // update weights and biases
    auto update(uint32_t batchsize) -> ErrorType override
    {

        if (m_ptrWeightGradient == nullptr || m_ptrBiasGradient == nullptr) {
            return ErrorType::UnknownError;
        }

        // Loop through all filters (output channels)
        for (int oc = 0; oc < m_header->channelsout; ++oc) {
            // Loop through all input channels
            for (int ic = 0; ic < m_header->channelsin; ++ic) {
                for (int ky = 0; ky < m_header->kernelsize[0]; ++ky) {
                    for (int kx = 0; kx < m_header->kernelsize[1]; ++kx) {

                        // Linear index calculation for 4D tensor stored flat
                        int index = (((oc * m_header->channelsin + ic) * m_header->kernelsize[0] + ky) * m_header->kernelsize[1] + kx);

                        m_ptrWeight->update(index,
                            m_ptrWeightGradient[index] / batchsize,
                            this->m_model->m_learningRate,
                            m_timestep);
                    }
                }
            }
            // Bias-Update: 1 value per filter
            m_ptrBias->update(oc,
                m_ptrBiasGradient[oc] / batchsize,
                this->m_model->m_learningRate,
                m_timestep);
        }
        m_timestep++;
        return ErrorType::OK;
    }

    // Loads the trainable values from Flash into SRAM
    auto loadFromFlash() -> ErrorType override
    {
        switch (m_optimizerType) {
        case OptimizerID::SGD:
            // init weights
            m_ptrWeight = new OptimizerSGD<T>;
            m_ptrWeight->init(m_header->weights_amount_trainable);
            // init bias
            m_ptrBias = new OptimizerSGD<T>;
            m_ptrBias->init(m_header->bias_amount_trainable);
            break;

        case OptimizerID::Momentum:
            // init weights
            m_ptrWeight = new OptimizerMomentum<T>;
            m_ptrWeight->init(m_header->weights_amount_trainable);
            // init bias
            m_ptrBias = new OptimizerMomentum<T>;
            m_ptrBias->init(m_header->bias_amount_trainable);
            break;

        case OptimizerID::ADAM:
            // init weights
            m_ptrWeight = new OptimizerAdam<T>;
            m_ptrWeight->init(m_header->weights_amount_trainable);
            // init bias
            m_ptrBias = new OptimizerAdam<T>;
            m_ptrBias->init(m_header->bias_amount_trainable);
            break;
        }

        // init weights and biases
        for (size_t i = 0; i < m_header->weights_amount_trainable; i++) {
            m_ptrWeight->setData(i, *(static_cast<T*>(m_ptrData) + (m_header->weights_trainable_offset / sizeof(T)) + i));
        }
        for (size_t i = 0; i < m_header->bias_amount_trainable; i++) {
            m_ptrBias->setData(i, *(static_cast<T*>(m_ptrData) + (m_header->bias_trainable_offset / sizeof(T)) + i));
        }

        this->m_isLoaded = true;
        return ErrorType::OK;
    }

    // Stores the trainable values from SRAM to Flash
    auto storeToFlash() -> ErrorType override
    {
        this->m_isLoaded = false;
        // Not implemented in this version, will only become relevant on the uController
        return ErrorType::UnknownError;
    }

    auto getOutputSize() -> uint32_t override
    {
        return uint32_t(m_header->dimensionoutput_x * m_header->dimensionoutput_y * m_header->channelsout);
    }

    auto getInputSize() -> uint32_t override
    {
        return uint32_t(m_header->dimensioninput_x * m_header->dimensioninput_y * m_header->channelsin);
    }

    [[nodiscard]] auto header() const -> const Neural_Network_Conv2d_t&
    {
        return *m_header;
    }

private:
    void* m_ptrLayer;
    void* m_ptrData;
    Neural_Network_Conv2d_t* m_header;

    // chosen optimizer
    OptimizerID m_optimizerType;

    T* m_ptrWeightFrozen;
    T* m_ptrBiasFrozen;

    T* m_ptrFlashWeight;
    T* m_ptrFlashBias;

    T* m_ptrWeightGradient = nullptr;
    T* m_ptrBiasGradient = nullptr;

    // vector with output Values
    T* m_layerOutput;

    // vector with the trainable weights in SRAM
    OptimizerBase<T>* m_ptrWeight;
    OptimizerBase<T>* m_ptrBias;

    uint32_t m_timestep = 1;
};
} // namespace Edgeist

#endif // CONV_2D_H
