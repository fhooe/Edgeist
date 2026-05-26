/**
 * @file
 * @author David Muttenthaler
 * @brief Defines base and derived classes for neural network optimizers
 */

#ifndef EDGEIST_OPTIMIZER_DATA_TYPES
#define EDGEIST_OPTIMIZER_DATA_TYPES

#include <cmath>
#include <cstdint>
#include <vector>

#include "nmcf_error_types.h"

namespace Edgeist {
enum class OptimizerID : std::uint8_t {
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
	
	std::vector<T> m_data;
};

template <typename T>
class OptimizerSGD : public OptimizerBase<T> {
public:
    //size_t m_dataSize = 0;

    OptimizerSGD() = default;
    ~OptimizerSGD() override
    {
    }

    T getData(uint32_t index) const override
    {
        if (index >= this->m_data.size()) {
            return T(0);
        }
        return this->m_data[index];
    }

    T getM(uint32_t /* index */) const override
    {
        return T(0);
    }

    T getV(uint32_t /* index */) const override
    {
        return T(0);
    }

    ErrorType setData(uint32_t index, T value) override
    {
        if (index >= this->m_data.size()) {
            return ErrorType::IndexOutOfBounds;
        }
        this->m_data[index] = value;
        return ErrorType::OK;
    }

    ErrorType setM(uint32_t /* index */, T /* value */) override
    {
        return ErrorType::OK;
    }

    ErrorType setV(uint32_t /* index */, T /* value */) override
    {
        return ErrorType::OK;
    }

    ErrorType init(uint32_t size) override
    {
	    this->m_data.resize(size);
        
        for (size_t i = 0; i < size; i++) {
            this->m_data[i] = T(0.0);
        }

        return ErrorType::OK;
    }

    ErrorType update(uint32_t index, T gradient, T learningRate, uint32_t /* timestep = 1 */) override
    {
        if (index >= this->m_data.size()) {
            return ErrorType::IndexOutOfBounds;
        }

        this->m_data[index] -= learningRate * gradient;
        return ErrorType::OK;
    }
};

template <typename T>
class OptimizerMomentum : public OptimizerBase<T> {
public:
    //T* m_M = nullptr;
	std::vector<T> m_M;

    //size_t m_dataSize = 0;

    static constexpr T BETA = T(0.9);

    OptimizerMomentum() = default;
    ~OptimizerMomentum() override
    {

       // if (this->m_data != nullptr) {
       //     delete[] this->m_data;
       //     this->m_data = nullptr;
       // }

       // if (m_M != nullptr) {
       //     delete[] m_M;
       //     m_M = nullptr;
       // }
    }

    T getData(uint32_t index) const override
    {
        if (index >= this->m_data.size()) {
            return T(0);
        }
        return this->m_data[index];
    }

    T getM(uint32_t index) const override
    {
        if (index >= this->m_M.size()) {
            return T(0);
        }
        return m_M[index];
    }

    T getV(uint32_t /* index */) const override
    {
        return T(0);
    }

    ErrorType setData(uint32_t index, T value) override
    {
        if (index >= this->m_data.size()) {
            return ErrorType::IndexOutOfBounds;
        }
        this->m_data[index] = value;
        return ErrorType::OK;
    }

    ErrorType setM(uint32_t index, T value) override
    {
        if (index >= m_M.size()) {
            return ErrorType::IndexOutOfBounds;
        }
        m_M[index] = value;
        return ErrorType::OK;
    }

    ErrorType setV(uint32_t /* index */, T /* value */) override
    {
        return ErrorType::OK;
    }

    ErrorType init(uint32_t size) override
    {
        this->m_data.resize(size);
        m_M.resize(size);

        for (size_t i = 0; i < size; i++) {
            this->m_data[i] = T(0.0);
            m_M[i] = T(0.0);
        }

        return ErrorType::OK;
    }

    ErrorType update(uint32_t index, T gradient, T learningRate, uint32_t /* timestep = 1 */) override
    {
        if (index >= m_M.size()) {
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
   // T* m_M = nullptr;
   // T* m_V = nullptr;

std::vector<T> m_M;
std::vector<T> m_V;

//    size_t m_dataSize = 0;

    static constexpr T BETA1 = T(0.9);
    static constexpr T BETA2 = T(0.999);
    static constexpr T EPSILON = T(1e-8);

    OptimizerAdam() = default;
    ~OptimizerAdam() override = default;

    T getData(uint32_t index) const override
    {
        return this->m_data[index];
    }

    T getM(uint32_t index) const override
    {
        if (index >= m_M.size()) {
            return T(0);
        }
        return m_M[index];
    }

    T getV(uint32_t index) const override
    {
        if (index >= m_V.size()) {
            return T(0);
        }
        return m_V[index];
    }

    ErrorType setData(uint32_t index, T value) override
    {
        if (index >= this->m_data.size()) {
            return ErrorType::IndexOutOfBounds;
        }
        this->m_data[index] = value;
        return ErrorType::OK;
    }

    ErrorType setM(uint32_t index, T value) override
    {
        if (index >= m_M.size()) {
            return ErrorType::IndexOutOfBounds;
        }
        m_M[index] = value;
        return ErrorType::OK;
    }

    ErrorType setV(uint32_t index, T value) override
    {
        if (index >= m_V.size()) {
            return ErrorType::IndexOutOfBounds;
        }
        m_V[index] = value;
        return ErrorType::OK;
    }

    ErrorType init(uint32_t size) override
    {
        this->m_data.resize(size);
	m_M.resize(size);
	m_V.resize(size);

        for (size_t i = 0; i < size; i++) {
            this->m_data[i] = T(0.0);
            m_M[i] = T(0.0);
            m_V[i] = T(0.0);
        }

        return ErrorType::OK;
    }

    ErrorType update(uint32_t index, T gradient, T learningRate, uint32_t timestep) override
    {
        if (index >= m_M.size()) {
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

#endif // EDGEIST_OPTIMIZER_DATA_TYPES
