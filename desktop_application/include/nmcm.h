/**
 * @file
 * @author David Muttenthaler
 * @brief Implements a template-based neural network model
 */

#ifndef NMCM_H
#define NMCM_H

#include <iostream>
#include <memory>
#include <vector>

#include "Modelenums.h"
#include "Modeltypes.h"
#include "adaptive_avg_pool_1d.h"
#include "adaptive_avg_pool_2d.h"
#include "batch_norm_1d.h"
#include "batch_norm_2d.h"
#include "conv_2d.h"
#include "dropout.h"
#include "flatten.h"
#include "layer.h"
#include "linear.h"
#include "loss_function.h"
#include "max_pool_2d.h"
#include "nmcf_error_types.h"
#include "object.h"
#include "optimizer_data_types.h"
#include "relu.h"
#include "softmax.h"

namespace Edgeist {
using com = size_t;

/**
 * @brief Template-based implementation of a neural network model
 *
 * This class encapsulates all key operations of a neural network, including:
 *  - Initialization from flash memory
 *  - Forward pass (inference)
 *  - Backward pass (training)
 *  - Weight loading and saving
 *  - Support for distributed learning (using partial weight blocks)
 *
 * The model architecture is defined in flash and initialized into SRAM at runtime.
 * Layers are dynamically instantiated based on their LayerID values.
 *
 * To generate a compatible network, use the python NMCF-parser
 * to convert a (pretrained) pytorch model into the right format.
 * @tparam T Datatype of the neural network inputs.
 */
template <typename T>
class Model : public Object {
public:
    // init model
    auto init() -> ErrorType
    {
        // Generate Layers
        for (int i = 0; i < m_header->layernrs; i++) {
            // GenerateLayer(layerPointers[i], )
            auto layerId = *m_ptrLayerPointers[i];

            // Print Layers with IDs
            std::cout << "Layer: " << i << "; LayerID: " << std::to_string(layerId) << std::endl;

            switch (static_cast<Nmcm::LayerId>(layerId)) {
            case Nmcm::LayerId::Linear:
                m_layersInSRAM.push_back(std::make_shared<Linear<T>>(this, getLayerPtr(i), m_ptrData, m_optimizerType));
                break;

            case Nmcm::LayerId::ReLU:
                m_layersInSRAM.push_back(std::make_shared<Relu<T>>(this, getLayerPtr(i), m_ptrData, m_optimizerType));
                break;

            case Nmcm::LayerId::Softmax:
                m_layersInSRAM.push_back(std::make_shared<Softmax<T>>(this, getLayerPtr(i), m_ptrData, m_optimizerType));
                break;

            case Nmcm::LayerId::Conv2d:
                m_layersInSRAM.push_back(std::make_shared<Conv2D<T>>(this, getLayerPtr(i), m_ptrData, m_optimizerType));
                break;

            case Nmcm::LayerId::Flatten:
                m_layersInSRAM.push_back(std::make_shared<Flatten<T>>(this, getLayerPtr(i), m_ptrData, m_optimizerType));
                break;

            case Nmcm::LayerId::MaxPool2d:
                m_layersInSRAM.push_back(std::make_shared<MaxPool2D<T>>(this, getLayerPtr(i), m_ptrData, m_optimizerType));
                break;

            case Nmcm::LayerId::BatchNorm1d:
                m_layersInSRAM.push_back(std::make_shared<BatchNorm1D<T>>(this, getLayerPtr(i), m_ptrData, m_optimizerType));
                break;

            case Nmcm::LayerId::BatchNorm2d:
                m_layersInSRAM.push_back(std::make_shared<BatchNorm2D<T>>(this, getLayerPtr(i), m_ptrData, m_optimizerType));
                break;

            case Nmcm::LayerId::AdaptiveAvgPool1d:
                m_layersInSRAM.push_back(std::make_shared<AdaptiveAvgPool1D<T>>(this, getLayerPtr(i), m_ptrData, m_optimizerType));
                break;

            case Nmcm::LayerId::AdaptiveAvgPool2d:
                m_layersInSRAM.push_back(std::make_shared<AdaptiveAvgPool2D<T>>(this, getLayerPtr(i), m_ptrData, m_optimizerType));
                break;

            case Nmcm::LayerId::Dropout:
                m_layersInSRAM.push_back(std::make_shared<Dropout<T>>(this, getLayerPtr(i), m_ptrData, m_optimizerType));
                break;

            default:
                break;
            }
        }

        return ErrorType::OK;
    }

    // Run Inference from Flash
    auto inferenceFlash(const T* input, T* output) -> ErrorType
    {
        return inferenceSram(input, output, false);
    }

    // Run Inference from Sram
    auto inferenceSram(const T* input, T* output, bool trainingFlag = true) -> ErrorType
    {
        m_ptrInputData = input;
        m_ptrOutputData = output;

        T* ptrLayerOutputData = nullptr;
        T* ptrLayerInputData = nullptr;

        // Find largest needed Buffer
        size_t buffersize = 0;
        for (int i = 0; i < m_header->layernrs; i++) {
            size_t size = m_layersInSRAM[i]->getOutputSize();
            if (size > buffersize) {
                buffersize = size;
            }
        }

        // allocate buffers
        ptrLayerOutputData = new T[buffersize];
        ptrLayerInputData = new T[buffersize];

        // Make a forward Pass through all the Layers
        for (int i = 0; i < m_header->layernrs; i++) {
            // First Layer gets input from Method argument
            if (i == 0) {
                m_layersInSRAM[i]->forwardPass(m_ptrInputData, ptrLayerOutputData, trainingFlag);
            }
            // Other Layers get Input from Previous Layer
            else {

                m_layersInSRAM[i]->forwardPass(ptrLayerInputData, ptrLayerOutputData, trainingFlag);
            }

            T* ptrswap = ptrLayerInputData;
            ptrLayerInputData = ptrLayerOutputData;
            ptrLayerOutputData = ptrswap;
        }

        for (size_t i = 0; i < m_header->dimensionoutput_x; i++) {
            // the last output was moved to the input ptr, copy the values to the output registers
            m_ptrOutputData[i] = ptrLayerInputData[i];
        }

        delete[] ptrLayerInputData;
        ptrLayerInputData = nullptr;
        delete[] ptrLayerOutputData;
        ptrLayerOutputData = nullptr;

        return ErrorType::OK;
    }

    // Train Method
    auto train(const T* input, T* expectedOutput, const LossFunction<T>& lossFn) -> ErrorType
    {
        mPtrExpectedOutputData = expectedOutput;

        m_ptrInputData = input;

        T* ptrLayerOutputData = nullptr;
        T* ptrLayerInputData = nullptr;

        // Find largest needed Buffer
        size_t buffersize = 0;
        for (int i = 0; i < m_header->layernrs; i++) {
            size_t size = m_layersInSRAM[i]->getOutputSize();
            if (size > buffersize) {
                buffersize = size;
            }
        }

        // allocate buffers
        ptrLayerOutputData = new T[buffersize];
        ptrLayerInputData = new T[buffersize];

        // Make a forward Pass through all the Layers
        for (int i = 0; i < m_header->layernrs; i++) {
            // First Layer gets input as argument
            if (i == 0) {
                m_layersInSRAM[i]->forwardPass(m_ptrInputData, ptrLayerOutputData, true);
            } else {
                // Other Layers get Input from Previous Layer
                m_layersInSRAM[i]->forwardPass(ptrLayerInputData, ptrLayerOutputData, true);
            }

            T* ptrswap = ptrLayerInputData;
            ptrLayerInputData = ptrLayerOutputData;
            ptrLayerOutputData = ptrswap;
        }

        T* ptrGradient = new T[(m_header->dimensionoutput_x)];

        lossFn.derivative(ptrLayerInputData, mPtrExpectedOutputData, ptrGradient, m_header->dimensionoutput_x);

        // Make a backward Pass through all the Layers
        for (int i = m_header->layernrs - 1; i >= 0; i--) {

            // Last Layer gets expected Output as argument
            if (i == m_header->layernrs - 1) {
                m_layersInSRAM[i]->backwardPass(ptrGradient, ptrLayerOutputData);
            }
            // Other Layers get Input from Previous Layer
            else {
                m_layersInSRAM[i]->backwardPass(ptrLayerInputData, ptrLayerOutputData);
            }

            T* ptrswap = ptrLayerInputData;
            ptrLayerInputData = ptrLayerOutputData;
            ptrLayerOutputData = ptrswap;
        }

        for (size_t i = 0; i < m_header->dimensionoutput_x; i++) {
            // the last output was moved to the input ptr, copy the values to the output registers
            m_ptrOutputData[i] = ptrLayerInputData[i];
        }

        delete[] ptrGradient;
        ptrGradient = nullptr;
        delete[] ptrLayerInputData;
        ptrLayerInputData = nullptr;
        delete[] ptrLayerOutputData;
        ptrLayerOutputData = nullptr;

        return ErrorType::OK;
    }

    auto initGradients() -> ErrorType
    {
        // Update all the Layers
        for (int i = 0; i < m_header->layernrs; i++) {
            m_layersInSRAM[i]->initGradients();
        }
        return ErrorType::OK;
    }

    auto deleteGradients() -> ErrorType
    {
        // Update all the Layers
        for (int i = 0; i < m_header->layernrs; i++) {
            m_layersInSRAM[i]->deleteGradients();
        }
        return ErrorType::OK;
    }

    auto update(uint32_t batchsize) -> ErrorType
    {
        // Update all the Layers
        for (int i = 0; i < m_header->layernrs; i++) {
            m_layersInSRAM[i]->update(batchsize);
        }
        return ErrorType::OK;
    }

    // access method for model header
    [[nodiscard]] auto header() const -> const Nmcm::Header::NeuralNetwork_t&
    {
        return *m_header;
    }

    // access method for Layer by index
    auto getLayer(uint32_t index) -> Layer<T>*
    {
        if (index >= m_layersInSRAM.size()) {
            return nullptr;
        }
        return m_layersInSRAM.at(index);
    }

    // access method to Input data by index
    auto getInput(uint32_t index) -> T
    {
        if (m_ptrInputData == nullptr || m_header == nullptr) {
            // Error no Input data or no Header
            return T(0);
        }

        uint32_t maxIndex = m_header->dimensioninput_x * m_header->dimensioninput_y * m_header->channelsin;
        if (index >= maxIndex) {
            // Error index out of bound
            return T(0);
        }

        return *(m_ptrInputData + index);
    }

    // Constructor
    Model(void* modelPointer, void* dataPointer, const OptimizerID optimizerType, const float learningRate)
        : m_learningRate(learningRate)
        , m_ptrModel(modelPointer)
        , m_ptrData(dataPointer)
        , m_optimizerType(optimizerType)
    {
        // populate the Header
        m_header = static_cast<Nmcm::Header::NeuralNetwork_t*>(m_ptrModel);

        // Initialize the layer pointer array with correct size
        m_ptrLayerPointers.resize(m_header->layernrs, nullptr);

        // populate the layer Pointer array
        for (int i = 0; i < m_header->layernrs; i++) {
            // Add offset to the pointer address of header
            size_t offset = (sizeof(Nmcm::Header::NeuralNetwork_t) - m_header->layernrs * 4) / sizeof(uint32_t) + i;
            uint32_t* targetAddressOffset = static_cast<uint32_t*>(m_ptrModel) + offset;

            m_ptrLayerPointers[i] = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(m_ptrModel) + *targetAddressOffset);
        }
    };

    float m_learningRate;

    // Pointer to expected Output Data
    T* mPtrExpectedOutputData = nullptr;

private:
    // Base Information from file
    Nmcm::Header::NeuralNetwork_t* m_header = nullptr;

    // pointer to the model in Flash
    void* m_ptrModel = nullptr;

    // pointer to the trainable data in Flash
    void* m_ptrData = nullptr;

    // Pointer to the Layers in Flash
    std::vector<std::shared_ptr<uint8_t>> m_ptrLayerPointers;

    // vector with the Layers in SRAM
    std::vector<std::shared_ptr<Layer<T>>> m_layersInSRAM;

    // Pointer to Input Data
    const T* m_ptrInputData = nullptr;

    // Pointer to Output Data
    T* m_ptrOutputData = nullptr;

    // init Flag, set when model.init is called and returns successfully.
    bool m_isInit = false;

    // Model generated Flag
    bool m_isLoaded = false;

    // Optimizer for the model
    OptimizerID m_optimizerType;

    // returns void Pointer to Layer by Index
    [[nodiscard]] auto getLayerPtr(const size_t layerNr) const -> void*
    {
        // Add offset to the pointer address of header
        size_t offset = (sizeof(Nmcm::Header::NeuralNetwork_t) - m_header->layernrs * 4) / sizeof(uint32_t) + layerNr;
        uint32_t* targetAddress = static_cast<uint32_t*>(m_ptrModel) + offset;

        return static_cast<void*>(static_cast<uint8_t*>(m_ptrModel) + *targetAddress);
    }
};
} // namespace Edgeist

#endif // NMCM_H
