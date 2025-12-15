/**
 * @author David Muttenthaler
 * @brief Defines base and derived classes for neural network optimizers.
 **/

#ifndef OPTIMIZER_DATA_TYPES
#define OPTIMIZER_DATA_TYPES

#include "nmcf_error_types.h"
#include <cmath>

namespace Edgeist {
enum OptimizerID {
    SGD = 0,
    Momentum = 1,
    ADAM = 2
};

template <typename T>
class OptimizerBase {
public:
    virtual auto getData(uint32_t index) const -> T = 0;
    virtual auto getM(uint32_t index) const -> T = 0;
    virtual auto getV(uint32_t index) const -> T = 0;

    virtual auto setData(uint32_t index, T value) -> ErrorType = 0;
    virtual auto setM(uint32_t index, T value) -> ErrorType = 0;
    virtual auto setV(uint32_t index, T value) -> ErrorType = 0;

    virtual auto init(uint32_t size) -> ErrorType = 0;

    virtual auto update(uint32_t index, T gradient, T learningRate, uint32_t timestep = 1) -> ErrorType = 0;

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

    auto getData(uint32_t index) const -> T override
    {
        if (index >= mData_size) {
            return T(0);
        }
        return this->mData[index];
    }

    auto getM(uint32_t /* index */) const -> T override
    {
        return T(0);
    }

    auto getV(uint32_t /* index */) const -> T override
    {
        return T(0);
    }

    auto setData(uint32_t index, T value) -> ErrorType override
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        this->mData[index] = value;
        return ErrorType::ok;
    }

    auto setM(uint32_t /* index */, T /* value */) -> ErrorType override
    {
        return ErrorType::ok;
    }

    auto setV(uint32_t /* index */, T /* value */) -> ErrorType override
    {
        return ErrorType::ok;
    }

    auto init(uint32_t size) -> ErrorType override
    {

        this->mData = new T[size];
        mData_size = size;
        for (size_t i = 0; i < size; i++) {
            this->mData[i] = T(0.0);
        }

        return ErrorType::ok;
    }

    auto update(uint32_t index, T gradient, T learningRate, uint32_t /* timestep = 1 */) -> ErrorType override
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

    auto getData(uint32_t index) const -> T override
    {
        if (index >= mData_size) {
            return T(0);
        }
        return this->mData[index];
    }

    auto getM(uint32_t index) const -> T override
    {
        if (index >= mData_size) {
            return T(0);
        }
        return mM[index];
    }

    auto getV(uint32_t /* index */) const -> T override
    {
        return T(0);
    }

    auto setData(uint32_t index, T value) -> ErrorType override
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        this->mData[index] = value;
        return ErrorType::ok;
    }

    auto setM(uint32_t index, T value) -> ErrorType override
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        mM[index] = value;
        return ErrorType::ok;
    }

    auto setV(uint32_t /* index */, T /* value */) -> ErrorType override
    {
        return ErrorType::ok;
    }

    auto init(uint32_t size) -> ErrorType override
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

    auto update(uint32_t index, T gradient, T learningRate, uint32_t /* timestep = 1 */) -> ErrorType override
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

    auto getData(uint32_t index) const -> T override
    {
        return this->mData[index];
    }

    auto getM(uint32_t index) const -> T override
    {
        if (index >= mData_size) {
            return T(0);
        }
        return mM[index];
    }

    auto getV(uint32_t index) const -> T override
    {
        if (index >= mData_size) {
            return T(0);
        }
        return mV[index];
    }

    auto setData(uint32_t index, T value) -> ErrorType override
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        this->mData[index] = value;
        return ErrorType::ok;
    }

    auto setM(uint32_t index, T value) -> ErrorType override
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        mM[index] = value;
        return ErrorType::ok;
    }

    auto setV(uint32_t index, T value) -> ErrorType override
    {
        if (index >= mData_size) {
            return ErrorType::IndexOutOfBounds;
        }
        mV[index] = value;
        return ErrorType::ok;
    }

    auto init(uint32_t size) -> ErrorType override
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

    auto update(uint32_t index, T gradient, T learningRate, uint32_t timestep) -> ErrorType override
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
} // namespace Edgeist

#endif // OPTIMIZER_DATA_TYPES
