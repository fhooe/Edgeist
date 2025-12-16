/**
 * @file
 * @author David Muttenthaler
 * @brief Defines base and derived classes for neural network optimizers.
 */

#ifndef OPTIMIZER_DATA_TYPES
#define OPTIMIZER_DATA_TYPES

#include "nmcf_error_types.h"
#include <cmath>

namespace Edgeist {
enum class OptimizerID {
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

    T* m_data = nullptr;
};

template <typename T>
class OptimizerSGD : public OptimizerBase<T> {
public:
    size_t m_dataSize = 0;

    OptimizerSGD() = default;
    ~OptimizerSGD() override
    {
        if (this->m_data != nullptr) {
            delete[] this->m_data;
            this->m_data = nullptr;
        }
    }

    auto getData(uint32_t index) const -> T override
    {
        if (index >= m_dataSize) {
            return T(0);
        }
        return this->m_data[index];
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
        if (index >= m_dataSize) {
            return ErrorType::IndexOutOfBounds;
        }
        this->m_data[index] = value;
        return ErrorType::OK;
    }

    auto setM(uint32_t /* index */, T /* value */) -> ErrorType override
    {
        return ErrorType::OK;
    }

    auto setV(uint32_t /* index */, T /* value */) -> ErrorType override
    {
        return ErrorType::OK;
    }

    auto init(uint32_t size) -> ErrorType override
    {
        this->m_data = new T[size];
        m_dataSize = size;
        for (size_t i = 0; i < size; i++) {
            this->m_data[i] = T(0.0);
        }

        return ErrorType::OK;
    }

    auto update(uint32_t index, T gradient, T learningRate, uint32_t /* timestep = 1 */) -> ErrorType override
    {
        if (index >= m_dataSize) {
            return ErrorType::IndexOutOfBounds;
        }

        this->m_data[index] -= learningRate * gradient;
        return ErrorType::OK;
    }
};

template <typename T>
class OptimizerMomentum : public OptimizerBase<T> {
public:
    T* m_M = nullptr;

    size_t m_dataSize = 0;

    static constexpr T BETA = T(0.9);

    OptimizerMomentum() = default;
    ~OptimizerMomentum() override
    {

        if (this->m_data != nullptr) {
            delete[] this->m_data;
            this->m_data = nullptr;
        }

        if (m_M != nullptr) {
            delete[] m_M;
            m_M = nullptr;
        }
    }

    auto getData(uint32_t index) const -> T override
    {
        if (index >= m_dataSize) {
            return T(0);
        }
        return this->m_data[index];
    }

    auto getM(uint32_t index) const -> T override
    {
        if (index >= m_dataSize) {
            return T(0);
        }
        return m_M[index];
    }

    auto getV(uint32_t /* index */) const -> T override
    {
        return T(0);
    }

    auto setData(uint32_t index, T value) -> ErrorType override
    {
        if (index >= m_dataSize) {
            return ErrorType::IndexOutOfBounds;
        }
        this->m_data[index] = value;
        return ErrorType::OK;
    }

    auto setM(uint32_t index, T value) -> ErrorType override
    {
        if (index >= m_dataSize) {
            return ErrorType::IndexOutOfBounds;
        }
        m_M[index] = value;
        return ErrorType::OK;
    }

    auto setV(uint32_t /* index */, T /* value */) -> ErrorType override
    {
        return ErrorType::OK;
    }

    auto init(uint32_t size) -> ErrorType override
    {
        this->m_data = new T[size];
        m_M = new T[size];

        m_dataSize = size;
        for (size_t i = 0; i < size; i++) {
            this->m_data[i] = T(0.0);
            m_M[i] = T(0.0);
        }

        return ErrorType::OK;
    }

    auto update(uint32_t index, T gradient, T learningRate, uint32_t /* timestep = 1 */) -> ErrorType override
    {
        if (index >= m_dataSize) {
            return ErrorType::IndexOutOfBounds;
        }

        m_M[index] = BETA * m_M[index] + (1 - BETA) * gradient;
        this->m_data[index] -= learningRate * m_M[index];
        return ErrorType::OK;
    }
};

template <typename T>
class OptimizerAdam : public OptimizerBase<T> {
public:
    T* m_M = nullptr;
    T* m_V = nullptr;

    size_t m_dataSize = 0;

    static constexpr T BETA1 = T(0.9);
    static constexpr T BETA2 = T(0.999);
    static constexpr T EPSILON = T(1e-8);

    OptimizerAdam() = default;
    ~OptimizerAdam() override = default;

    auto getData(uint32_t index) const -> T override
    {
        return this->m_data[index];
    }

    auto getM(uint32_t index) const -> T override
    {
        if (index >= m_dataSize) {
            return T(0);
        }
        return m_M[index];
    }

    auto getV(uint32_t index) const -> T override
    {
        if (index >= m_dataSize) {
            return T(0);
        }
        return m_V[index];
    }

    auto setData(uint32_t index, T value) -> ErrorType override
    {
        if (index >= m_dataSize) {
            return ErrorType::IndexOutOfBounds;
        }
        this->m_data[index] = value;
        return ErrorType::OK;
    }

    auto setM(uint32_t index, T value) -> ErrorType override
    {
        if (index >= m_dataSize) {
            return ErrorType::IndexOutOfBounds;
        }
        m_M[index] = value;
        return ErrorType::OK;
    }

    auto setV(uint32_t index, T value) -> ErrorType override
    {
        if (index >= m_dataSize) {
            return ErrorType::IndexOutOfBounds;
        }
        m_V[index] = value;
        return ErrorType::OK;
    }

    auto init(uint32_t size) -> ErrorType override
    {
        this->m_data = new T[size];
        m_M = new T[size];
        m_V = new T[size];

        m_dataSize = size;

        for (size_t i = 0; i < size; i++) {
            this->m_data[i] = T(0.0);
            m_M[i] = T(0.0);
            m_V[i] = T(0.0);
        }

        return ErrorType::OK;
    }

    auto update(uint32_t index, T gradient, T learningRate, uint32_t timestep) -> ErrorType override
    {
        if (index >= m_dataSize) {
            return ErrorType::IndexOutOfBounds;
        }

        // calculate m and v
        m_M[index] = BETA1 * m_M[index] + (1 - BETA1) * gradient;
        m_V[index] = BETA2 * m_V[index] + (1 - BETA2) * gradient * gradient;

        // bias correction
        T mHat = m_M[index] / (1 - std::pow(BETA1, timestep));
        T vHat = m_V[index] / (1 - std::pow(BETA2, timestep));

        // update
        this->m_data[index] -= learningRate * mHat / (std::sqrt(vHat) + EPSILON);
        return ErrorType::OK;
    }
};
} // namespace Edgeist

#endif // OPTIMIZER_DATA_TYPES
