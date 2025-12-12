#ifndef OPTIMIZERDATATYPES
#define OPTIMIZERDATATYPES

#include "nmcf_error_types.h"
#include <cmath>
#include <vector>

/**
 * @author David Muttenthaler
 * @date 23-06-2025
 *
 * @file OptimizerDataTypes.h
 * @brief Defines base and derived classes for neural network optimizers.
 *
 * This header provides a polymorphic interface (`OptimizerBase`) for various optimization
 * algorithms used during neural network training. Each optimizer manages its own internal
 * state and weight update logic based on gradient descent methods.
 *
 * Supported optimizers include:
 *  - SGD (Stochastic Gradient Descent)
 *  - Momentum
 *  - Adam (Adaptive Moment Estimation)
 *
 * All optimizers are templated to support different numeric types (e.g., float, double).
 * Memory management is handled manually to ensure compatibility with embedded systems or
 * low-level runtime environments.
 *
 * The design allows for dynamic optimizer selection using the `OptimizerID` enum and runtime
 * polymorphism through the `OptimizerBase` interface.
 *
 */

enum OptimizerID {
    SGD = 0,
    Momentum = 1,
    ADAM = 2
};

template <typename T>
class OptimizerBase {
public:
    virtual T getData(uint32_t index) const = 0;
    virtual T getM(uint32_t index) const = 0;
    virtual T getV(uint32_t index) const = 0;

    virtual ErrorType setData(uint32_t index, T value) = 0;
    virtual ErrorType setM(uint32_t index, T value) = 0;
    virtual ErrorType setV(uint32_t index, T value) = 0;

    virtual ErrorType init(uint32_t size) = 0;

    virtual ErrorType update(uint32_t index, T gradient, T learningRate, uint32_t timestep = 1) = 0;

    virtual ~OptimizerBase() = default;

    T* mData = nullptr;
};

template <typename T>
class OptimizerSGD : public OptimizerBase<T> {
public:
    size_t mData_size = 0;

    OptimizerSGD() = default;
    ~OptimizerSGD() override
    {
        if (this->mData != nullptr) {
            delete[] this->mData;
            this->mData = nullptr;
        }
    }

    T getData(uint32_t index) const
    {
        if (index >= mData_size) {
            return T(0);
        }
        return this->mData[index];
    }

    T getM(uint32_t /* index */) const
    {
        return T(0);
    }

    T getV(uint32_t /* index */) const
    {
        return T(0);
    }

    ErrorType setData(uint32_t index, T value)
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        this->mData[index] = value;
        return ErrorType::ok;
    }

    ErrorType setM(uint32_t /* index */, T /* value */)
    {
        return ErrorType::ok;
    }

    ErrorType setV(uint32_t /* index */, T /* value */)
    {
        return ErrorType::ok;
    }

    virtual ErrorType init(uint32_t size)
    {

        this->mData = new T[size];
        mData_size = size;
        for (size_t i = 0; i < size; i++) {
            this->mData[i] = T(0.0);
        }

        return ErrorType::ok;
    }

    ErrorType update(uint32_t index, T gradient, T learningRate, uint32_t /* timestep = 1 */) override
    {
        if (index >= mData_size)
            return ErrorType::IndexOutOfBounds;

        this->mData[index] -= learningRate * gradient;
        return ErrorType::ok;
    }
};

template <typename T>
class OptimizerMomentum : public OptimizerBase<T> {
public:
    T* mM = nullptr;

    size_t mData_size = 0;

    T beta = T(0.9);

    OptimizerMomentum() = default;
    ~OptimizerMomentum() override
    {

        if (this->mData != nullptr) {
            delete[] this->mData;
            this->mData = nullptr;
        }

        if (mM != nullptr) {
            delete[] mM;
            mM = nullptr;
        }
    }

    T getData(uint32_t index) const
    {
        if (index >= mData_size) {
            return T(0);
        }
        return this->mData[index];
    }

    T getM(uint32_t index) const
    {
        if (index >= mData_size) {
            return T(0);
        }
        return mM[index];
    }

    T getV(uint32_t /* index */) const
    {
        return T(0);
    }

    ErrorType setData(uint32_t index, T value)
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        this->mData[index] = value;
        return ErrorType::ok;
    }

    ErrorType setM(uint32_t index, T value)
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        mM[index] = value;
        return ErrorType::ok;
    }

    ErrorType setV(uint32_t /* index */, T /* value */)
    {
        return ErrorType::ok;
    }

    virtual ErrorType init(uint32_t size)
    {

        this->mData = new T[size];
        mM = new T[size];

        mData_size = size;
        for (size_t i = 0; i < size; i++) {
            this->mData[i] = T(0.0);
            mM[i] = T(0.0);
        }

        return ErrorType::ok;
    }

    ErrorType update(uint32_t index, T gradient, T learningRate, uint32_t /* timestep = 1 */) override
    {
        if (index >= mData_size)
            return ErrorType::IndexOutOfBounds;

        mM[index] = beta * mM[index] + (1 - beta) * gradient;
        this->mData[index] -= learningRate * mM[index];
        return ErrorType::ok;
    }
};

template <typename T>
class OptimizerAdam : public OptimizerBase<T> {
public:
    T* mM = nullptr;
    T* mV = nullptr;

    size_t mData_size = 0;

    T beta1 = T(0.9);
    T beta2 = T(0.999);
    T epsilon = T(1e-8);

    OptimizerAdam() = default;
    ~OptimizerAdam() override = default;

    T getData(uint32_t index) const
    {
        return this->mData[index];
    }

    T getM(uint32_t index) const
    {
        if (index >= mData_size) {
            return T(0);
        }
        return mM[index];
    }

    T getV(uint32_t index) const
    {
        if (index >= mData_size) {
            return T(0);
        }
        return mV[index];
    }

    ErrorType setData(uint32_t index, T value)
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        this->mData[index] = value;
        return ErrorType::ok;
    }

    ErrorType setM(uint32_t index, T value)
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        mM[index] = value;
        return ErrorType::ok;
    }

    ErrorType setV(uint32_t index, T value)
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        mV[index] = value;
        return ErrorType::ok;
    }

    virtual ErrorType init(uint32_t size)
    {

        this->mData = new T[size];
        mM = new T[size];
        mV = new T[size];

        mData_size = size;

        for (size_t i = 0; i < size; i++) {
            this->mData[i] = T(0.0);
            mM[i] = T(0.0);
            mV[i] = T(0.0);
        }

        return ErrorType::ok;
    }

    ErrorType update(uint32_t index, T gradient, T learningRate, uint32_t timestep) override
    {
        if (index >= mData_size)
            return ErrorType::IndexOutOfBounds;

        // m und v berechnen
        mM[index] = beta1 * mM[index] + (1 - beta1) * gradient;
        mV[index] = beta2 * mV[index] + (1 - beta2) * gradient * gradient;

        // Bias-Korrektur
        T m_hat = mM[index] / (1 - std::pow(beta1, timestep));
        T v_hat = mV[index] / (1 - std::pow(beta2, timestep));

        // Update
        this->mData[index] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        return ErrorType::ok;
    }
};

#endif // !OPTIMIZERDATATYPES
