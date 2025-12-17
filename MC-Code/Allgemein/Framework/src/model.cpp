#include "pch.h"
#include "model.h"
#include "Socket_setup.h"

template<typename T>
ErrorType model<T>::GenerateModel(int ** Layers)
{
	return ErrorType::ok;
}

template<typename T>
ErrorType model<T>::GenerateModel(int ** Layers, size_t Start, size_t Stop)
{
	return ErrorType::ok;
}

template<typename T>
ErrorType model<T>::TransferWeights(size_t Start, size_t Stop, com_t next)
{
	return ErrorType::ok;
}

template<typename T>
ErrorType model<T>::SelectClass(size_t id, int * layeraddress)
{
	return ErrorType::ok;
}


template<typename T>
ErrorType model<T>::InferenceFlash(T * input, T * output)
{
	return ErrorType::ok;
}


template<typename T>
ErrorType model<T>::InferenceSRAM(T * input, T * output, size_t Start, size_t Stop, com_t next)
{
	return ErrorType::ok;
}

template<typename T>
ErrorType model<T>::saveWeights()
{
	return ErrorType::ok;
}

