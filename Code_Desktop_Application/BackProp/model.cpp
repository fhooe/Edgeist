#include "pch.h"
#include "model.h"

template<typename T>
ErrorType model<T>::GenerateModel(int ** Layers)
{
	return None;
}

template<typename T>
ErrorType model<T>::GenerateModel(int ** Layers, size_t Start, size_t Stop)
{
	return None;
}

template<typename T>
ErrorType model<T>::TransferWeights(size_t Start, size_t Stop, com next)
{
	return None;
}

template<typename T>
ErrorType model<T>::SelectClass(size_t id, int * layeraddress)
{
	return None;
}


template<typename T>
ErrorType model<T>::InferenceFlash(T * input, T * output)
{
	return None;
}

template<typename T>
ErrorType model<T>::InferenceSRAM(T * input, T * output)
{
	return None;
}

template<typename T>
ErrorType model<T>::InferenceSRAM(T * input, T * output, size_t Start, size_t Stop, com next)
{
	return None;
}

template<typename T>
ErrorType model<T>::Train(T * input, T * expectedOutput)
{
	return None;
}

template<typename T>
ErrorType model<T>::Train(T * input, T * expectedOutput, size_t Start, size_t Stop, com next)
{
	return None;
}

template<typename T>
ErrorType model<T>::loadWeights()
{
	return None;
}

template<typename T>
ErrorType model<T>::loadWeights(size_t Start, size_t Stop)
{
	return None;
}

template<typename T>
ErrorType model<T>::saveWeights()
{
	return None;
}

template<typename T>
ErrorType model<T>::saveWeights(size_t Start, size_t Stop)
{
	return None;
}
