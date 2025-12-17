#pragma once
#include <vector>
#include <iostream>

#include "object.h"
#include "ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include "Modeltypes.h"
#include "Modelenums.h"
#include "Layer.h"
#include "Linear.h"
#include "Relu.h"
#include "Conf2d.h"
#include "Softmax.h"
#include "Flatten.h"
#include "MaxPool2d.h"

#include "Socket_setup.h"
#include "Protocol.h"

typedef struct
{
	ip_t* IP_Addresses;
	uint8_t size;
	uint8_t pos;
}com_t;

template <typename T>
class model :
	public object
{
public:
	// init model
	ErrorType Init() {

		// Generate Layers
		for (int i = 0; i < mHeader->layernrs; i++) {
			//GenerateLayer(layerPointers[i], )
			int ID = static_cast<int>(*mPtrLayerPointers[i]);

			//Print Layers with IDs
			std::cout << "Layer: " << i << " LayerID: " << ID << std::endl;

			switch (ID) {
				case int(LayerIDs::Linear_ID) :
					layersInSRAM.push_back(new Linear<T>(this, getLayerPtr(i), mPtrData, mOptimizerType));
					break;

				case int(LayerIDs::ReLU_ID) :
					layersInSRAM.push_back(new Relu<T>(this, getLayerPtr(i), mPtrData, mOptimizerType));
					break;

				case int(LayerIDs::Softmax_ID) :
					layersInSRAM.push_back(new Softmax<T>(this, getLayerPtr(i), mPtrData, mOptimizerType));
					break;

				case int(LayerIDs::Conv2d_ID) :
					layersInSRAM.push_back(new Conv2d<T>(this, getLayerPtr(i), mPtrData, mOptimizerType));
					break;

				case int(LayerIDs::Flatten_ID) :
					layersInSRAM.push_back(new Flatten<T>(this, getLayerPtr(i), mPtrData, mOptimizerType));
					break;

				case int(LayerIDs::MaxPool2d_ID) :
					layersInSRAM.push_back(new MaxPool2d<T>(this, getLayerPtr(i), mPtrData, mOptimizerType));
					break;

				default:
					break;
			}
		}

		return ErrorType::ok;
	}

	// Run Inference from Flash
	ErrorType InferenceFlash(T* input, T* output);

	// Run Inference from Sram
	ErrorType InferenceSRAM(const T* input, T* output) {
		mPtrInputData = input;
		mPtrOutputData = output;

		T* ptrLayerOutputData = nullptr;
		T* ptrLayerInputData = nullptr;


		//TODO remove -1 and implement a flatten layer
		// Make a forward Pass throu all the Layers
		for (int i = 0; i < mHeader->layernrs; i++) {

			//allocate array for Output of the current Layer
			T* ptrLayerOutputData = new T[layersInSRAM[i]->getOutputSize()];

			// First Layer gets input from Methode argumet
			if (i == 0) {
				if (layersInSRAM[i]->forwardPass(mPtrInputData, ptrLayerOutputData) != ErrorType::ok)
				{
					while(1);
				}
			}
			//Other Layers get Input from Previous Layer
			else {

				if (layersInSRAM[i]->forwardPass(ptrLayerInputData, ptrLayerOutputData) != ErrorType::ok)
				{
					while(1);
				}
			}

			//l�sche den alten Input Buffer
			delete[] ptrLayerInputData;

			//letzter Output wird zum neuen input. Output ptr wird gel�scht
			ptrLayerInputData = ptrLayerOutputData;
			ptrLayerOutputData = nullptr;
		}



		for (int i = 0; i < mHeader->dimensionoutput_x; i++) {
			// the last output was moved to the input ptr, copy the values to the output registers
			mPtrOutputData[i] = ptrLayerInputData[i];
		}

		//free memory
		delete[] ptrLayerInputData; ptrLayerInputData = nullptr;


		return ErrorType::ok;
	}


	// Run Inference from Sram
	ErrorType InferenceSRAM(T* input, T* output, uint32_t Start, uint32_t Stop, com_t next);


	// Train Methode
	ErrorType Train(const T* input, T* expectedOutput) {

		mPtrExpectedOutputData = expectedOutput;

		mPtrInputData = input;

		T* ptrLayerOutputData = nullptr;
		T* ptrLayerInputData = nullptr;



		// Make a forward Pass through all the Layers
		for (int i = 0; i < mHeader->layernrs; i++) {

			//allocate array for Output of the current Layer
			uint32_t length = layersInSRAM[i]->getOutputSize();
			T* ptrLayerOutputData = new T[length];

			// First Layer gets input as argumet
			if (i == 0) {
				layersInSRAM[i]->forwardPass(mPtrInputData, ptrLayerOutputData, true);
			}
			//Other Layers get Input from Previous Layer
			else {

				layersInSRAM[i]->forwardPass(ptrLayerInputData, ptrLayerOutputData, true);
			}

			//loesche den alten Input Buffer
			delete[] ptrLayerInputData;

			//letzter Output wird zum neuen input. Output ptr wird geloescht
			ptrLayerInputData = ptrLayerOutputData; ptrLayerOutputData = nullptr;
		}


		// Make a backward Pass through all the Layers
		for (int i = mHeader->layernrs - 1; i > 0; i--) {

			//allocate array for Output of the current Layer
			int lenght = layersInSRAM[i]->getInputSize();
			T* ptrLayerOutputData = new T[lenght];

			// Last Layer gets expected Output as argumet
			if (i == mHeader->layernrs - 1) {
				layersInSRAM[i]->backwardPass(ptrLayerInputData, ptrLayerOutputData);
			}
			//Other Layers get Input from Previous Layer
			else {

				layersInSRAM[i]->backwardPass(ptrLayerInputData, ptrLayerOutputData);
			}

			//loesche den alten Input Buffer
			delete[] ptrLayerInputData;

			//letzter Output wird zum neuen input. Output ptr wird geloescht
			ptrLayerInputData = ptrLayerOutputData; ptrLayerOutputData = nullptr;
		}



		for (int i = 0; i < mHeader->dimensionoutput_x; i++) {
			// the last output was moved to the input ptr, copy the values to the output registers
			mPtrOutputData[i] = ptrLayerInputData[i];
		}

		//free memory
		delete[] ptrLayerInputData; ptrLayerInputData = nullptr;


		return ErrorType::ok;

	}
	// Train methode for distributed Learning
	ErrorType Train(const T* input, T* expectedOutput, uint32_t Start, uint32_t Stop, com_t* CommunicationInfo)
	{
		mPtrExpectedOutputData = expectedOutput;
		mPtrInputData = input;

		T* ptrLayerOutputData = nullptr;
		T* ptrLayerInputData = nullptr;

		if (Stop > mHeader->layernrs)
		{
			return ErrorType::IndexOutOfBounds;
		}

		if (Start >= Stop)
		{
			return ErrorType::IndexOutOfBounds;
		}

		// Make a forward Pass through some the Layers
		for (int i = Start; i < Stop; i++)
		{
			//allocate array for Output of the current Layer
			T* ptrLayerOutputData = new T[layersInSRAM[i]->getOutputSize()];

			// First Layer gets input as argumet
			if (i == Start) {
				layersInSRAM[i]->forwardPass(mPtrInputData, ptrLayerOutputData, true);
			}
			//Other Layers get Input from Previous Layer
			else {
				layersInSRAM[i]->forwardPass(ptrLayerInputData, ptrLayerOutputData, true);
			}

			//loesche den alten Input Buffer
			delete[] ptrLayerInputData;

			//letzter Output wird zum neuen input. Output ptr wird geloescht
			ptrLayerInputData = ptrLayerOutputData; 
			ptrLayerOutputData = nullptr;
		}

		// transmit data to the next device and wait for calculated data
		if ((CommunicationInfo->pos + 1) != CommunicationInfo->size)
		{
			// Send Output of the last layer to the next device
			Protocol instruction = Protocol::Train;
			send_to_direct(CommunicationInfo->IP_Addresses[CommunicationInfo->pos+1],reinterpret_cast<uint8_t*>(&instruction), sizeof(Protocol));
			
			// send Forward Pass Output
			uint32_t length = layersInSRAM[Stop-1]->getOutputSize() * sizeof(T); 
			send_to_direct(CommunicationInfo->IP_Addresses[CommunicationInfo->pos+1],reinterpret_cast<uint8_t*>(&length), sizeof(uint32_t));
			send_to_direct(CommunicationInfo->IP_Addresses[CommunicationInfo->pos+1],reinterpret_cast<uint8_t*>(ptrLayerInputData),length);

			// send expected Output
			send_to_direct(CommunicationInfo->IP_Addresses[CommunicationInfo->pos+1],reinterpret_cast<uint8_t*>(expectedOutput), layersInSRAM[mHeader->layernrs-1]->getOutputSize() * sizeof(T));
			
			
			// Receive the backward pass output from the previous device
			int32_t retLength = 0;
			while (retLength != sizeof(uint32_t))
			{
				retLength = recv_from_direct(CommunicationInfo->IP_Addresses[CommunicationInfo->pos+1], reinterpret_cast<uint8_t*>(&length), sizeof(uint32_t));
				if (retLength < 0)
				{
					return ErrorType::UnknownError;
				}
			}
			while (retLength != length)
			{
				retLength = recv_from_direct(CommunicationInfo->IP_Addresses[CommunicationInfo->pos+1],reinterpret_cast<uint8_t*>(ptrLayerInputData),length);
				if (retLength < 0)
				{
					return ErrorType::UnknownError;
				}
			}
		}

		// Make a backward Pass through some the Layers
		for (int i = Stop - 1; i >= (int)Start; i--) {

			//allocate array for Output of the current Layer
			int lenght = layersInSRAM[i]->getInputSize();
			T* ptrLayerOutputData = new T[lenght];

			// Last Layer gets expected Output as argumet
			if (i == mHeader->layernrs - 1) {
				layersInSRAM[i]->backwardPass(ptrLayerInputData, ptrLayerOutputData);
			}
			//Other Layers get Input from Previous Layer
			else {

				layersInSRAM[i]->backwardPass(ptrLayerInputData, ptrLayerOutputData);
			}

			//loesche den alten Input Buffer
			delete[] ptrLayerInputData;

			//letzter Output wird zum neuen input. Output ptr wird geloescht
			ptrLayerInputData = ptrLayerOutputData; ptrLayerOutputData = nullptr;
		}

		if (Start == 0)
		{
			for (int i = 0; i < mHeader->dimensionoutput_x; i++) {
				// the last output was moved to the input ptr, copy the values to the output registers
				mPtrOutputData[i] = ptrLayerInputData[i];
			}
		}
		else
		{
			// send output of backward Pass to previous Layer
			uint32_t length = layersInSRAM[Start]->getInputSize() * sizeof(T); 
			send_to_direct(CommunicationInfo->IP_Addresses[CommunicationInfo->pos-1],reinterpret_cast<uint8_t*>(&length), sizeof(uint32_t));
			send_to_direct(CommunicationInfo->IP_Addresses[CommunicationInfo->pos-1],reinterpret_cast<uint8_t*>(ptrLayerInputData),length);
		}

		//free memory
		delete[] ptrLayerInputData; ptrLayerInputData = nullptr;

		return ErrorType::ok;
	}

	ErrorType initGradients() {
		// Update all the Layers
		for (int i = 0; i < mHeader->layernrs; i++) {
			layersInSRAM[i]->initGradients();
		}
		return ErrorType::ok;
	}

	ErrorType initGradients(uint32_t Start, uint32_t Stop) {
		if ((Stop < Start) || (Stop > mHeader->layernrs))
		{
			return ErrorType::IndexOutOfBounds;
		}
		// Update all the local Layers
		for (int i = Start; i < Stop; i++) {
			layersInSRAM[i]->initGradients();
		}
		return ErrorType::ok;
	}
	
	ErrorType deleteGradients() {
		// Delete all Gradients
		for (int i = 0; i < mHeader->layernrs; i++) {
			layersInSRAM[i]->deleteGradients();
		}
		return ErrorType::ok;
	}

	ErrorType deleteGradients(uint32_t Start, uint32_t Stop) {
		if ((Stop < Start) || (Stop > mHeader->layernrs))
		{
			return ErrorType::IndexOutOfBounds;
		}
		// Delete all local Gradients
		for (int i = Start; i < Stop; i++) {
			layersInSRAM[i]->deleteGradients();
		}
		return ErrorType::ok;
	}

	ErrorType Update(uint32_t batchsize) {
		// Update all the Layers
		for (int i = 0; i < mHeader->layernrs; i++) {
			layersInSRAM[i]->update(batchsize);
		}
		return ErrorType::ok;
	}

	ErrorType Update(uint32_t batchsize, uint32_t Start, uint32_t Stop) {
		if ((Stop < Start) || (Stop > mHeader->layernrs))
		{
			return ErrorType::IndexOutOfBounds;
		}
		// Update all the Layers
		for (int i = Start; i < Stop; i++) {
			layersInSRAM[i]->update(batchsize);
		}
		return ErrorType::ok;
	}
	
	ErrorType UpdateFlash(com_t &CommunicationData, uint8_t* buffer, uint32_t BUFFER_SIZE, Flash_manager &fm, uint32_t* start_data, uint32_t Start, uint32_t Stop, bool Firstdevice = false) 
	{
		// Continuously receive and send data until a stop signal is received from the final device
		uint8_t EndSwap = 1;
		uint32_t Weigths_offset = 0;
		uint32_t Bias_offset = 0;
		
		int32_t bytes_received = 0;
		uint32_t length_to_recv = 0;
		uint32_t bytes_send = 0;
		int32_t retval = 0;
		int32_t ip_pos = -1;
		int32_t idx = -1;
		uint8_t nAll_written = 1;
		uint32_t size_buffer = 0;		// is used to write information at Flashsync, data is not needed
		
		uint8_t* nFlash_written = new uint8_t[CommunicationData.size];
		
		Protocol instruction = Firstdevice ? Protocol::New_Sender : Protocol::None;
		Layerinformation_t layer_info;
		
		if (Stop > mHeader->layernrs)
		{
			return ErrorType::IndexOutOfBounds;
		}

		if (Start >= Stop)
		{
			return ErrorType::IndexOutOfBounds;
		}

		while(EndSwap)
		{
			if (Firstdevice == false)
			{
				// get Instruction
				ip_pos = listen_all(reinterpret_cast<uint8_t*>(&instruction),sizeof(Protocol));
				if (ip_pos < 0)
				{
					continue;
				}
			}
			
			switch (instruction)
			{
				case Protocol::Finished_Write_Flash:
					EndSwap = 0;
					break;
				
				case Protocol::Write_Flash:
					// in this case the device recvies data from other devices and stores it in it's flash
				
					// get layerinformation
					bytes_received = 0;
					while (bytes_received != sizeof(layer_info))
					{
						bytes_received = recv_from_direct(CommunicationData.IP_Addresses[ip_pos],reinterpret_cast<uint8_t*>(&layer_info),sizeof(layer_info));
						if (bytes_received < 0)
						{
							while(1);
						}
					}
					
					// get weigths (BUFFER_SIZE needs to be a multible of Tx-buffer of Transimttion)
					bytes_received = 0;
					length_to_recv = layer_info.Weigthamount * sizeof(float);
					while (bytes_received != layer_info.Weigthamount * sizeof(float))
					{
						if (length_to_recv > BUFFER_SIZE)
						{
							retval = recv_from_direct(CommunicationData.IP_Addresses[ip_pos],buffer,BUFFER_SIZE);
							if (retval < 0)
							{
								while(1);
							}

							length_to_recv -= retval;
						}
						else
						{
							retval = recv_from_direct(CommunicationData.IP_Addresses[ip_pos],buffer,length_to_recv);
							if (retval < 0)
							{
								while(1);
							}
						}
						
						// write recvied data from buffer into flash into flash
						this->saveExternalData(&fm, start_data, bytes_received, layer_info.LayerNumber, (uint32_t*)buffer, (retval >> 2));
						
						// send the Transmiter that the data was written into the flash
						if (retval != 0)
						{
							send_to_direct(CommunicationData.IP_Addresses[ip_pos], reinterpret_cast<uint8_t*>(&size_buffer), sizeof(size_buffer));
						}
						
						bytes_received += retval;
					}
					
					// get bias (BUFFER_SIZE needs to be a multible of Tx-buffer of Transimttion)
					length_to_recv = layer_info.Biasamount * sizeof(float);
					while (bytes_received != (layer_info.Weigthamount + layer_info.Biasamount) * sizeof(float))
					{
						if (length_to_recv > BUFFER_SIZE)
						{
							retval = recv_from_direct(CommunicationData.IP_Addresses[ip_pos],buffer,BUFFER_SIZE);
							if (retval < 0)
							{
								while(1);
							}
							length_to_recv -= BUFFER_SIZE;
						}
						else
						{
							retval = recv_from_direct(CommunicationData.IP_Addresses[ip_pos],buffer,length_to_recv);
							if (retval < 0)
							{
								while(1);
							}
						}
						
						// write recvied data from buffer into flash into flash
						this->saveExternalData(&fm, start_data, bytes_received, layer_info.LayerNumber, (uint32_t*)buffer, (retval >> 2));
						
						// send the Transmiter that the data was written into the flash
						if (retval != 0)
						{
							send_to_direct(CommunicationData.IP_Addresses[ip_pos], reinterpret_cast<uint8_t*>(&size_buffer), sizeof(size_buffer));
						}
						
						bytes_received += retval;
					}
					break;
				
				case Protocol::New_Sender:
					// in this case the device sends it data to all the other
					Firstdevice = false;
				
					for (uint32_t i = Start; i < Stop; i++)
					{
						instruction = Protocol::Write_Flash;
						broadcast_direct(reinterpret_cast<uint8_t*>(&instruction), sizeof(Protocol));
						
						// send layer info
						layer_info.LayerNumber = i;
						this->getLayer(i)->getWeigths(Weigths_offset, layer_info.Weigthamount);
						this->getLayer(i)->getBias(Bias_offset, layer_info.Biasamount);
						broadcast_direct(reinterpret_cast<uint8_t*>(&layer_info),sizeof(layer_info));
						
						bytes_send = 0;
						while (bytes_send != layer_info.Weigthamount * sizeof(float))
						{
							if (bytes_send + BUFFER_SIZE < layer_info.Weigthamount * sizeof(float))
							{
								broadcast_direct((uint8_t*)start_data + Weigths_offset + bytes_send, BUFFER_SIZE);
								bytes_send += BUFFER_SIZE;
							}
							else
							{
								// last transmittion
								broadcast_direct((uint8_t*)start_data + Weigths_offset + bytes_send, layer_info.Weigthamount * sizeof(float) - bytes_send);
								bytes_send += layer_info.Weigthamount * sizeof(float) - bytes_send;
							}
							
							// wait for all devices to write to flash
							nAll_written = 1;
							memset(nFlash_written, 1, CommunicationData.size);
							nFlash_written[CommunicationData.pos] = 0;
							while (nAll_written)
							{
								idx = listen_all(reinterpret_cast<uint8_t*>(&size_buffer),sizeof(size_buffer));
								if (idx < 0)
								{
									continue;
								}
								if (idx >= CommunicationData.size)
								{
									while(1);
								}
								
								nFlash_written[idx] = 0;
								
								nAll_written = 0;
								for (uint8_t i = 0; i < CommunicationData.size; i++)
								{
									nAll_written |= nFlash_written[i];
								}
							}
						}
						
						bytes_send = 0;
						while (bytes_send != layer_info.Biasamount * sizeof(float))
						{
							if (bytes_send + BUFFER_SIZE < layer_info.Biasamount * sizeof(float))
							{
								broadcast_direct((uint8_t*)start_data + Bias_offset + bytes_send, BUFFER_SIZE);
								bytes_send += BUFFER_SIZE;
							}
							else
							{
								// last transmittion
								broadcast_direct((uint8_t*)start_data + Bias_offset + bytes_send, layer_info.Biasamount * sizeof(float) - bytes_send);
								bytes_send += layer_info.Biasamount * sizeof(float) - bytes_send;
							}
							
							// wait for all devices to write to flash
							nAll_written = 1;
							memset(nFlash_written, 1, CommunicationData.size);
							nFlash_written[CommunicationData.pos] = 0;
							while (nAll_written)
							{
								idx = listen_all(reinterpret_cast<uint8_t*>(&size_buffer),sizeof(size_buffer));
								if (idx < 0)
								{
									continue;
								}
								if (idx >= CommunicationData.size)
								{
									while(1);
								}
								
								nFlash_written[idx] = 0;
								
								nAll_written = 0;
								for (uint8_t i = 0; i < CommunicationData.size; i++)
								{
									nAll_written |= nFlash_written[i];
								}
							}
							
						}
					}
				
					// check if update process is finished
					if (CommunicationData.pos + 1 == CommunicationData.size)
					{
						// finished
						instruction = Protocol::Finished_Write_Flash;
						broadcast_direct(reinterpret_cast<uint8_t*>(&instruction), sizeof(Protocol));
						EndSwap = 0;
					}
					else
					{
						// next device sends data
						instruction = Protocol::New_Sender;
						send_to_direct(CommunicationData.IP_Addresses[CommunicationData.pos+1],reinterpret_cast<uint8_t*>(&instruction), sizeof(Protocol));
					}
					
				default:
					break;
			}
		}
		return ErrorType::ok;
	}

	// Load Weights from Flash to SRAM
	ErrorType loadWeights()
	{
		for (uint32_t i = 0; i < mHeader->layernrs; i++)
		{
			layersInSRAM[i]->loadFromFlash();
		}

		return ErrorType::ok;
	}

	// Load Weights methode for distributed Learning
	ErrorType loadWeights(uint32_t Start, uint32_t Stop)
	{
		if (Stop > mHeader->layernrs)
		{
			return ErrorType::IndexOutOfBounds;
		}

		if (Start >= Stop)
		{
			return ErrorType::IndexOutOfBounds;
		}

		for (uint32_t i = Start; i < Stop; i++)
		{
			layersInSRAM[i]->loadFromFlash();
		}

		return ErrorType::ok;
	}

	// Save Weights back to Flash from SRAM
	ErrorType saveWeights();
	
	// Sava Weights methode for distributed Learning
	ErrorType saveWeights(Flash_manager* fm, uint32_t* start_address, uint32_t Start, uint32_t Stop)
	{
		// write own layers in flash
		for (uint32_t i = Start; i < Stop; i++)
		{
			layersInSRAM[i]->storeToFlash(fm, start_address);
		}
		
		return ErrorType::ok;
	}
	
	// Write Weigths and Bias from other devices to flash
	// !!! doesn't clear the flash
	// Parameter:
	// 	fm = object to interact with the flash
	//	start_address = Address where all trainable Data starts
	// 	Layer_offset = Offset in one Layer
	// 	LayerID = ID of the Layer to update
	// 	Start_at_begin:
	//		true  : use start of Layer as 0
	// 		false : use bias of Layer as 0 
	ErrorType saveExternalData(Flash_manager* fm, uint32_t* start_address, uint32_t Layer_offset, uint32_t LayerID, uint32_t* data, uint32_t datasize, bool Start_at_begin = true)
	{
		if (LayerID >= layersInSRAM.size())
		{
			return ErrorType::IndexOutOfBounds;
		}
	
		if (datasize == 0)
		{
			return ErrorType::ok;
		}

		if (layersInSRAM.at(LayerID)->storeExternalToFlash(fm, (uint32_t)start_address + Layer_offset, data, datasize, Start_at_begin) != ErrorType::ok)
		{
			return ErrorType::UnknownError;
		}
		
		return ErrorType::ok;
	}

	// access methode vor model header
	const Neural_Network_Header_t& header() {
		return *mHeader;
	}

	// access methode for Layer by index
	Layer<T>* getLayer(uint32_t index) {

		if (index >= layersInSRAM.size())
		{
			return nullptr;
		}
		return layersInSRAM.at(index);
	}

	//access methode to Input data by index
	T getInput(uint32_t index) {

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
	model(void* ModelPointer, void* DataPointer, OptimizerID OptimizerType, float LearningRate) : mPtrModel(ModelPointer), mPtrData(DataPointer), mOptimizerType(OptimizerType), mLearningRate(LearningRate) {
		// populate the Header
		mHeader = static_cast<Neural_Network_Header_t*>(mPtrModel);

		// Initialize the layer pointer array with correct size
		mPtrLayerPointers.resize(mHeader->layernrs, nullptr);

		// populate the layer Pointer array
		for (int i = 0; i < mHeader->layernrs; i++) {
			size_t offset = (sizeof(Neural_Network_Header_t) - mHeader->layernrs*4) / sizeof(uint32_t) + i;
			uint32_t* targetAddressOffset = static_cast<uint32_t*>(mPtrModel) + offset;

			mPtrLayerPointers[i] = static_cast<uint8_t*>(mPtrModel) + *targetAddressOffset;
		}

		mExpectedOutputSize = mHeader->dimensionoutput_x * mHeader->dimensionoutput_y * mHeader->channelsout;
	};

	// Destructor
	~model() {

		for (Layer<T>* obj : layersInSRAM) {
			delete obj;
		}

	};

	float mLearningRate;

	// Pointer to expected Output Data
	T* mPtrExpectedOutputData = nullptr;
	
	uint32_t mExpectedOutputSize = 0;

private:

	// Base Information from file
	Neural_Network_Header_t* mHeader = nullptr;

	// pointer to the model in Flash
	void* mPtrModel = nullptr;

	// pointer to the trainable data in Flash
	void* mPtrData = nullptr;

	// Pointer to the Layers in Flash
	std::vector<uint8_t*> mPtrLayerPointers;

	// vector with the Layers in SRAM
	std::vector<Layer<T>*> layersInSRAM;

	// Pointer to Input Data
	const T* mPtrInputData = nullptr;

	// Pointer to Output Data
	T* mPtrOutputData = nullptr;


	// init Flag, set when model.init is called and returns succesfully.
	bool isInit = 0;

	// Activly training Flag
	bool isTraining = 0;

	// Model generated Flag
	bool mIsLoaded = 0;

	// Optimizer for the model
	OptimizerID mOptimizerType;

	// Generate model, sets up the necessary Layer Data and copies the trainable values to Ram and initializes the Optimizer Variables
	ErrorType GenerateModel(int** Layers);

	// Generate model for distributed Learning, sets up the necessary Layer Data and copies the trainable values to Ram and initializes the Optimizer Variables
	ErrorType GenerateModel(int** Layers, size_t Start, size_t Stop);

	// Transfer Weights, for distributed Learning
	ErrorType TransferWeights(size_t Start, size_t Stop, com_t next);

	// Why do we need that methode?
	ErrorType SelectClass(size_t id, int* layeraddress);

	// Private functions

	// returns void Pointer to Layer by Index
	void* getLayerPtr(size_t layerNr) {
		size_t offset = (sizeof(Neural_Network_Header_t) - mHeader->layernrs * 4) / sizeof(uint32_t) + layerNr;
		uint32_t* targetAddress = static_cast<uint32_t*>(mPtrModel) + offset;

		return static_cast<void*>(static_cast<uint8_t*>(mPtrModel) + *targetAddress);

	}

};

