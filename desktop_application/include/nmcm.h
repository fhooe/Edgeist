/**
 * @author David Muttenthaler
 * @brief Implements a template-based neural network model.
 **/

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
typedef size_t com;

/**
 * @brief Template-based implementation of a neural network model.
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
class model : public object {
public:
    // init model
    ErrorType Init()
    {

        // Generate Layers
        for (int i = 0; i < mHeader->layernrs; i++) {
            // GenerateLayer(layerPointers[i], )
            int ID = static_cast<int>(*mPtrLayerPointers[i]);

            // Print Layers with IDs
            std::cout << "Layer: " << i << "; LayerID: " << ID << std::endl;

            switch (ID) {
            case int(LayerIDs::Linear_ID):
                layersInSRAM.push_back(std::make_shared<Linear<T>>(this, getLayerPtr(i), mPtrData, mOptimizerType));
                break;

            case int(LayerIDs::ReLU_ID):
                layersInSRAM.push_back(std::make_shared<Relu<T>>(this, getLayerPtr(i), mPtrData, mOptimizerType));
                break;

            case int(LayerIDs::Softmax_ID):
                layersInSRAM.push_back(std::make_shared<Softmax<T>>(this, getLayerPtr(i), mPtrData, mOptimizerType));
                break;

            case int(LayerIDs::Conv2d_ID):
                layersInSRAM.push_back(std::make_shared<Conv2d<T>>(this, getLayerPtr(i), mPtrData, mOptimizerType));
                break;

            case int(LayerIDs::Flatten_ID):
                layersInSRAM.push_back(std::make_shared<Flatten<T>>(this, getLayerPtr(i), mPtrData, mOptimizerType));
                break;

            case int(LayerIDs::MaxPool2d_ID):
                layersInSRAM.push_back(std::make_shared<MaxPool2d<T>>(this, getLayerPtr(i), mPtrData, mOptimizerType));
                break;

            case int(LayerIDs::BatchNorm1d_ID):
                layersInSRAM.push_back(std::make_shared<BatchNorm1d<T>>(this, getLayerPtr(i), mPtrData, mOptimizerType));
                break;

            case int(LayerIDs::BatchNorm2d_ID):
                layersInSRAM.push_back(std::make_shared<BatchNorm2d<T>>(this, getLayerPtr(i), mPtrData, mOptimizerType));
                break;

            case int(LayerIDs::AdaptiveAvgPool1d_ID):
                layersInSRAM.push_back(std::make_shared<AdaptiveAvgPool1d<T>>(this, getLayerPtr(i), mPtrData, mOptimizerType));
                break;

            case int(LayerIDs::AdaptiveAvgPool2d_ID):
                layersInSRAM.push_back(std::make_shared<AdaptiveAvgPool2d<T>>(this, getLayerPtr(i), mPtrData, mOptimizerType));
                break;

            case int(LayerIDs::Dropout_ID):
                layersInSRAM.push_back(std::make_shared<Dropout<T>>(this, getLayerPtr(i), mPtrData, mOptimizerType));
                break;

            default:
                break;
            }
        }

        return ErrorType::ok;
    }

    // Run Inference from Flash
    ErrorType InferenceFlash(const T* input, T* output)
    {
        return InferenceSRAM(input, output, false);
    }

    // Run Inference from Sram
    ErrorType InferenceSRAM(const T* input, T* output, bool Trainingflag = true)
    {
        mPtrInputData = input;
        mPtrOutputData = output;

        T* ptrLayerOutputData = nullptr;
        T* ptrLayerInputData = nullptr;

        // Find lagest needed Buffer
        size_t buffersize = 0;
        for (int i = 0; i < mHeader->layernrs; i++) {
            size_t size = layersInSRAM[i]->getOutputSize();
            if (size > buffersize) {
                buffersize = size;
            }
        }

        // allocate buffers
        ptrLayerOutputData = new T[buffersize];
        ptrLayerInputData = new T[buffersize];

        // Make a forward Pass throu all the Layers
        for (int i = 0; i < mHeader->layernrs; i++) {

            // First Layer gets input from Methode argumet
            if (i == 0) {
                layersInSRAM[i]->forwardPass(mPtrInputData, ptrLayerOutputData, Trainingflag);
            }
            // Other Layers get Input from Previous Layer
            else {

                layersInSRAM[i]->forwardPass(ptrLayerInputData, ptrLayerOutputData, Trainingflag);
            }

            T* ptrswap = ptrLayerInputData;
            ptrLayerInputData = ptrLayerOutputData;
            ptrLayerOutputData = ptrswap;
        }

        for (size_t i = 0; i < mHeader->dimensionoutput_x; i++) {
            // the last output was moved to the input ptr, copy the values to the output registers
            mPtrOutputData[i] = ptrLayerInputData[i];
        }

        delete[] ptrLayerInputData;
        ptrLayerInputData = nullptr;
        delete[] ptrLayerOutputData;
        ptrLayerOutputData = nullptr;

        return ErrorType::ok;
    }

    // Run Inference from Sram
    ErrorType InferenceSRAM(T* input, T* output, uint32_t Start, uint32_t Stop, com next);

    // Train Methode
    ErrorType Train(const T* input, T* expectedOutput, const LossFunction<T>& lossFn)
    {

        mPtrExpectedOutputData = expectedOutput;

        mPtrInputData = input;

        T* ptrLayerOutputData = nullptr;
        T* ptrLayerInputData = nullptr;

        // Find lagest needed Buffer
        size_t buffersize = 0;
        for (int i = 0; i < mHeader->layernrs; i++) {
            size_t size = layersInSRAM[i]->getOutputSize();
            if (size > buffersize) {
                buffersize = size;
            }
        }

        // allocate buffers
        ptrLayerOutputData = new T[buffersize];
        ptrLayerInputData = new T[buffersize];

        // Make a forward Pass through all the Layers
        for (int i = 0; i < mHeader->layernrs; i++) {

            // First Layer gets input as argumet
            if (i == 0) {
                layersInSRAM[i]->forwardPass(mPtrInputData, ptrLayerOutputData, true);
            } else {
                // Other Layers get Input from Previous Layer
                layersInSRAM[i]->forwardPass(ptrLayerInputData, ptrLayerOutputData, true);
            }

            T* ptrswap = ptrLayerInputData;
            ptrLayerInputData = ptrLayerOutputData;
            ptrLayerOutputData = ptrswap;
        }

        T* ptrGradient = new T[(mHeader->dimensionoutput_x)];

        lossFn.derivative(ptrLayerInputData, mPtrExpectedOutputData, ptrGradient, mHeader->dimensionoutput_x);

        // Make a backward Pass through all the Layers
        for (int i = mHeader->layernrs - 1; i >= 0; i--) {

            // Last Layer gets expected Output as argumet
            if (i == mHeader->layernrs - 1) {
                layersInSRAM[i]->backwardPass(ptrGradient, ptrLayerOutputData);
            }
            // Other Layers get Input from Previous Layer
            else {

                layersInSRAM[i]->backwardPass(ptrLayerInputData, ptrLayerOutputData);
            }

            T* ptrswap = ptrLayerInputData;
            ptrLayerInputData = ptrLayerOutputData;
            ptrLayerOutputData = ptrswap;
        }

        for (size_t i = 0; i < mHeader->dimensionoutput_x; i++) {
            // the last output was moved to the input ptr, copy the values to the output registers
            mPtrOutputData[i] = ptrLayerInputData[i];
        }

        delete[] ptrGradient;
        ptrGradient = nullptr;
        delete[] ptrLayerInputData;
        ptrLayerInputData = nullptr;
        delete[] ptrLayerOutputData;
        ptrLayerOutputData = nullptr;

        return ErrorType::ok;
    }
    // Train methode for distributed Learning
    ErrorType Train(T* input, T* expectedOutput, uint32_t Start, uint32_t Stop, com next);

    ErrorType initGradients()
    {
        // Update all the Layers
        for (int i = 0; i < mHeader->layernrs; i++) {
            layersInSRAM[i]->initGradients();
        }
        return ErrorType::ok;
    }

    ErrorType deleteGradients()
    {
        // Update all the Layers
        for (int i = 0; i < mHeader->layernrs; i++) {
            layersInSRAM[i]->deleteGradients();
        }
        return ErrorType::ok;
    }

    ErrorType Update(uint32_t batchsize)
    {
        // Update all the Layers
        for (int i = 0; i < mHeader->layernrs; i++) {
            layersInSRAM[i]->update(batchsize);
        }
        return ErrorType::ok;
    }

    // Load Weights from Flash to SRAM
    ErrorType loadWeights();
    // Load Weights methode for distributed Learning
    ErrorType loadWeights(uint32_t Start, uint32_t Stop);

    // Save Weights back to Flash from SRAM
    ErrorType saveWeights();
    // Sava Weights methode for distributed Learning
    ErrorType saveWeights(uint32_t Start, uint32_t Stop);

    // access methode vor model header
    const Neural_Network_Header_t& header()
    {
        return *mHeader;
    }

    // access methode for Layer by index
    Layer<T>* getLayer(uint32_t index)
    {

        if (index >= layersInSRAM.size()) {
            return nullptr;
        }
        return layersInSRAM.at(index);
    }

    // access methode to Input data by index
    T getInput(uint32_t index)
    {

        if (mPtrInputData == nullptr || mHeader == nullptr) {
            // Error no Input data or no Header
            return T(0);
        }

        uint32_t maxIndex = mHeader->dimensioninput_x * mHeader->dimensioninput_y * mHeader->channelsin;
        if (index >= maxIndex) {
            // Error index out of bound
            return T(0);
        }

        return *(mPtrInputData + index);
    }

    // Constructor
    model(void* ModelPointer, void* DataPointer, OptimizerID OptimizerType, float LearningRate)
        : mLearningRate(LearningRate)
        , mPtrModel(ModelPointer)
        , mPtrData(DataPointer)
        , mOptimizerType(OptimizerType)
    {
        // populate the Header
        mHeader = static_cast<Neural_Network_Header_t*>(mPtrModel);

        // Initialize the layer pointer array with correct size
        mPtrLayerPointers.resize(mHeader->layernrs, nullptr);

        // populate the layer Pointer array
        for (int i = 0; i < mHeader->layernrs; i++) {
            // Add offset to the pointer address of header

            size_t offset = (sizeof(Neural_Network_Header_t) - mHeader->layernrs * 4) / sizeof(uint32_t) + i;
            uint32_t* targetAddressOffset = static_cast<uint32_t*>(mPtrModel) + offset;

            mPtrLayerPointers[i] = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(mPtrModel) + *targetAddressOffset);
        }
    };

    float mLearningRate;

    // Pointer to expected Output Data
    T* mPtrExpectedOutputData = nullptr;

private:
    // Base Information from file
    Neural_Network_Header_t* mHeader = nullptr;

    // pointer to the model in Flash
    void* mPtrModel = nullptr;

    // pointer to the trainable data in Flash
    void* mPtrData = nullptr;

    // Pointer to the Layers in Flash
    std::vector<std::shared_ptr<uint8_t>> mPtrLayerPointers;

    // vector with the Layers in SRAM
    std::vector<std::shared_ptr<Layer<T>>> layersInSRAM;

    // Pointer to Input Data
    const T* mPtrInputData = nullptr;

    // Pointer to Output Data
    T* mPtrOutputData = nullptr;

    // init Flag, set when model.init is called and returns succesfully.
    bool isInit = 0;

    // Model generated Flag
    bool mIsLoaded = 0;

    // Optimizer for the model
    OptimizerID mOptimizerType;

    // Generate model, sets up the necessary Layer Data and copies the trainable values to Ram and initializes the Optimizer Variables
    ErrorType GenerateModel(int** Layers);

    // Generate model for distributed Learning, sets up the necessary Layer Data and copies the trainable values to Ram and initializes the Optimizer Variables
    ErrorType GenerateModel(int** Layers, size_t Start, size_t Stop);

    // Transfer Weights, for distributed Learning
    ErrorType TransferWeights(size_t Start, size_t Stop, com next);

    // Why do we need that methode?
    ErrorType SelectClass(size_t id, int* layeraddress);

    // Private functions

    // returns void Pointer to Layer by Index
    void* getLayerPtr(size_t layerNr)
    {
        // Add offset to the pointer address of header
        size_t offset = (sizeof(Neural_Network_Header_t) - mHeader->layernrs * 4) / sizeof(uint32_t) + layerNr;
        uint32_t* targetAddress = static_cast<uint32_t*>(mPtrModel) + offset;

        return static_cast<void*>(static_cast<uint8_t*>(mPtrModel) + *targetAddress);
    }
};
} // namespace Edgeist

#endif // NMCM_H
