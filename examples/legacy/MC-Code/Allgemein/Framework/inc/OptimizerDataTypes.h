#ifndef OPTIMIZERDATATYPES
#define OPTIMIZERDATATYPES

#include <vector>
#include "ErrorTypes.h"

// Class for the Results of the NN and the Optimizer

enum OptimizerID {
	SGD = 0,
	Momentum = 1,
	ADAM = 2
};



template<typename T>
class OptimizerBase {
public:

	virtual T getData(uint32_t index) const = 0;
	virtual T const* getData() const = 0;
	virtual T getM(uint32_t index) const = 0;
	virtual T getV(uint32_t index) const = 0;

	virtual ErrorType setData(uint32_t index, T value) = 0;
	virtual ErrorType setM(uint32_t index, T value) = 0;
	virtual ErrorType setV(uint32_t index, T value) = 0;

	virtual ErrorType init(uint32_t size) = 0;

	virtual ErrorType update(uint32_t index, T gradient, T learningRate, uint32_t timestep = 1) = 0;

	virtual ~OptimizerBase() = default;
};

template<typename T>
class OptimizerSGD : public OptimizerBase<T> {
public:
	std::vector<T> mData;

	OptimizerSGD() = default;
	~OptimizerSGD() = default;

	T getData(uint32_t index) const override {
		if (index >= mData.size())
		{
			return T(0);
		}
		return mData.at(index);
	}

	T const* getData() const override {
		return mData.data();
	}
	
	T getM(uint32_t index) const override {
		return T(0);
	}

	T getV(uint32_t index) const override {
		return T(0);
	}

	ErrorType setData(uint32_t index, T value) override {
		if (index >= mData.size())
		{
			return ErrorType::IndexOutOfBounds;
		}
		mData.at(index) = value;
		return ErrorType::ok;
	}

	ErrorType setM(uint32_t index, T value) override {
		return ErrorType::ok;
	}

	ErrorType setV(uint32_t index, T value) override {
		return ErrorType::ok;
	}

	virtual ErrorType init(uint32_t size) override {

		mData = std::vector<T>(size, T(0));

		return ErrorType::ok;
	}

	virtual ErrorType update(uint32_t index, T gradient, T learningRate, uint32_t timestep = 1) override {
		if (index >= mData.size()) return ErrorType::IndexOutOfBounds;

		mData[index] -= learningRate * gradient;
		return ErrorType::ok;
	}
};

template<typename T>
class OptimizerMomentum : public OptimizerBase<T> {
public:
	std::vector<T> mData;
	std::vector<T> mM;

	T beta = T(0.9);


	OptimizerMomentum() = default;
	~OptimizerMomentum() = default;

	T getData(uint32_t index) const override {
		if (index >= mData.size())
		{
			return T(0);
		}
		return mData.at(index);
	}
	
	T const* getData() const override {
		return mData.data();
	}

	T getM(uint32_t index) const override {
		if (index >= mM.size())
		{
			return T(0);
		}
		return mM.at(index);
	}

	T getV(uint32_t index) const override {
		return T(0);
	}

	ErrorType setData(uint32_t index, T value) override {
		if (index >= mData.size())
		{
			return ErrorType::IndexOutOfBounds;
		}
		mData.at(index) = value;
		return ErrorType::ok;
	}

	ErrorType setM(uint32_t index, T value) override {
		if (index >= mM.size())
		{
			return ErrorType::IndexOutOfBounds;
		}
		mM.at(index) = value;
		return ErrorType::ok;
	}

	ErrorType setV(uint32_t index, T value) override {
		return ErrorType::ok;
	}

	virtual ErrorType init(uint32_t size) override {

		mData = std::vector<T>(size, T(0));
		mM = std::vector<T>(size, T(0));

		return ErrorType::ok;
	}

	virtual ErrorType update(uint32_t index, T gradient, T learningRate, uint32_t timestep = 1) override {
		if (index >= mData.size() || index >= mM.size()) return ErrorType::IndexOutOfBounds;

		mM[index] = beta * mM[index] + (1 - beta) * gradient;
		mData[index] -= learningRate * mM[index];
		return ErrorType::ok;
	}
};

template<typename T>
class OptimizerAdam : public OptimizerBase<T> {
public:
	std::vector<T> mData;
	std::vector<T> mM;
	std::vector<T> mV;

	T beta1 = T(0.9);
	T beta2 = T(0.999);
	T epsilon = T(1e-8);

	OptimizerAdam() = default;
	~OptimizerAdam() = default;

	T getData(uint32_t index) const override {
		if (index >= mData.size())
		{
			return T(0);
		}
		return mData.at(index);
	}

	T const* getData() const override {
		return mData.data();
	}
	
	T getM(uint32_t index) const override {
		if (index >= mM.size())
		{
			return T(0);
		}
		return mM.at(index);
	}

	T getV(uint32_t index) const override {
		if (index >= mV.size())
		{
			return T(0);
		}
		return mV.at(index);
	}

	ErrorType setData(uint32_t index, T value) override {
		if (index >= mData.size())
		{
			return ErrorType::IndexOutOfBounds;
		}
		mData.at(index) = value;
		return ErrorType::ok;
	}

	ErrorType setM(uint32_t index, T value) override {
		if (index >= mM.size())
		{
			return ErrorType::IndexOutOfBounds;
		}
		mM.at(index) = value;
		return ErrorType::ok;
	}

	ErrorType setV(uint32_t index, T value) override {
		if (index >= mV.size())
		{
			return ErrorType::IndexOutOfBounds;
		}
		mV.at(index) = value;
		return ErrorType::ok;
	}

	virtual ErrorType init(uint32_t size) override {

		mData = std::vector<T>(size, T(0));
		mM = std::vector<T>(size, T(0));
		mV = std::vector<T>(size, T(0));

		return ErrorType::ok;
	}

	virtual ErrorType update(uint32_t index, T gradient, T learningRate, uint32_t timestep) override {
		if (index >= mData.size() || index >= mM.size() || index >= mV.size()) return ErrorType::IndexOutOfBounds;

		// m und v berechnen
		mM[index] = beta1 * mM[index] + (1 - beta1) * gradient;
		mV[index] = beta2 * mV[index] + (1 - beta2) * gradient * gradient;

		// Bias-Korrektur
		T m_hat = mM[index] / (1 - std::pow(beta1, timestep));
		T v_hat = mV[index] / (1 - std::pow(beta2, timestep));

		// Update
		mData[index] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
		return ErrorType::ok;
	}


};


#endif // !OPTIMIZERDATATYPES
