/**
 * @file
 * @author David Muttenthaler
 * @brief Implements loss functions for neural networks.
 */

#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <cmath>

namespace Edgeist {
/**
 * @brief Linear loss function.
 *
 * Implements a generic linear loss function for neural networks.
 *
 * @tparam T Datatype of the loss functions inputs.
 */
template <typename T>
class LossFunction {
public:
    virtual ~LossFunction() = default;

    /**
     * @brief Calculates the loss between prediction and setpoint.
     *
     * @param predicted Pointer to array of prediction values.
     * @param target Pointer to array of setpoints (ground truth).
     * @param size Length of input arrays.
     * @return The calculated loss.
     */
    virtual T compute(const T* predicted, const T* target, std::size_t size) const = 0;

    /**
     * @brief Calculates the loss gradient for given prediction values.
     *
     * @param predicted Pointer to array of prediction values.
     * @param target Pointer to array of setpoints (ground truth).
     * @param output_grad Pointer to array storing the result of this method.
     * @param size Length of input arrays.
     */
    virtual void derivative(const T* predicted, const T* target, T* output_grad, std::size_t size) const = 0;
};

/**
 * @brief Mean Squared Error (MSE) loss function.
 *
 * This loss is commonly used for regression tasks. It computes the average
 * of the squared differences between predicted and target values.
 *
 * Pros:
 * - Smooth and differentiable.
 * - Penalizes larger errors more heavily.
 *
 * Cons:
 * - Sensitive to outliers due to squaring.
 *
 * @tparam T Datatype of the loss functions inputs.
 */
template <typename T>
class MSELoss : public LossFunction<T> {
public:
    T compute(const T* predicted, const T* target, std::size_t size) const override
    {
        T sum = 0;
        for (std::size_t i = 0; i < size; ++i) {
            T diff = predicted[i] - target[i];
            sum += diff * diff;
        }
        return sum / static_cast<T>(size);
    }

    void derivative(const T* predicted, const T* target, T* output_grad, std::size_t size) const override
    {
        for (std::size_t i = 0; i < size; ++i) {
            output_grad[i] = 2 * (predicted[i] - target[i]) / static_cast<T>(size);
        }
    }
};

/**
 * @brief Cross-Entropy loss function (without softmax).
 *
 * This loss is typically used for classification tasks with outputs that
 * already represent probabilities (e.g., after a sigmoid or softmax).
 *
 * Note: This version assumes inputs are already normalized to probabilities.
 *
 * @tparam T Datatype of the loss functions inputs.
 */
template <typename T>
class CrossEntropyLoss : public LossFunction<T> {
public:
    T compute(const T* predicted, const T* target, std::size_t size) const override
    {
        T loss = 0;
        for (std::size_t i = 0; i < size; ++i) {

            T p = std::max(predicted[i], static_cast<T>(1e-12));
            loss -= target[i] * std::log(p);
        }
        return loss;
    }

    void derivative(const T* predicted, const T* target, T* output_grad, std::size_t size) const override
    {
        for (std::size_t i = 0; i < size; ++i) {
            T p = std::max(predicted[i], static_cast<T>(1e-12));
            output_grad[i] = -target[i] / p;
        }
    }
};

/**
 * @brief Cross-Entropy loss combined with softmax activation (logits input).
 *
 * This is a numerically stable implementation commonly used for multi-class
 * classification problems. The softmax is applied internally on the raw logits.
 *
 * Gradient simplification:
 *     ∂L/∂logits = softmax(logits) - target
 *
 * @tparam T Datatype of the loss functions inputs.
 */
template <typename T>
class SoftmaxCrossEntropyLoss : public LossFunction<T> {
public:
    T compute(const T* logits, const T* target, std::size_t size) const override
    {

        T max_logit = logits[0];
        for (std::size_t i = 1; i < size; ++i)
            if (logits[i] > max_logit)
                max_logit = logits[i];

        T sum = 0;
        for (std::size_t i = 0; i < size; ++i)
            sum += std::exp(logits[i] - max_logit);

        T loss = 0;
        for (std::size_t i = 0; i < size; ++i) {
            T prob = std::exp(logits[i] - max_logit) / sum;
            T clipped = std::max(prob, static_cast<T>(1e-12));
            loss -= target[i] * std::log(clipped);
        }
        return loss;
    }

    void derivative(const T* logits, const T* target, T* output_grad, std::size_t size) const override
    {

        T max_logit = logits[0];
        for (std::size_t i = 1; i < size; ++i)
            if (logits[i] > max_logit)
                max_logit = logits[i];

        T sum = 0;
        for (std::size_t i = 0; i < size; ++i)
            sum += std::exp(logits[i] - max_logit);

        for (std::size_t i = 0; i < size; ++i) {
            T prob = std::exp(logits[i] - max_logit) / sum;
            output_grad[i] = prob - target[i];
        }
    }
};

/**
 * @brief Binary Cross-Entropy loss with integrated sigmoid activation.
 *
 * This version is used for binary classification when the model outputs logits
 * (i.e., no activation applied yet). Internally applies sigmoid and computes
 * BCE in a numerically stable way.
 *
 * Gradient simplification:
 *     ∂L/∂logits = sigmoid(logits) - target
 *
 * @tparam T Datatype of the loss functions inputs.
 */
template <typename T>
class SigmoidBinaryCrossEntropyLoss : public LossFunction<T> {
public:
    auto compute(const T* logits, const T* targets, std::size_t size) const -> T override
    {
        T loss = 0;
        for (std::size_t i = 0; i < size; ++i) {
            T z = logits[i];
            T y = targets[i];

            if (z >= 0) {
                loss += (std::log(1 + std::exp(-z)) + (1 - y) * z);
            } else {
                loss += (std::log(1 + std::exp(z)) - y * z);
            }
        }
        return loss / static_cast<T>(size);
    }

    auto derivative(const T* logits, const T* targets, T* output_grad, std::size_t size) const -> void override
    {
        for (std::size_t i = 0; i < size; ++i) {
            T z = logits[i];
            T y = targets[i];

            T sigmoid = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-z));
            output_grad[i] = (sigmoid - y) / static_cast<T>(size);
        }
    }
};
} // namespace Edgeist

#endif // LOSS_FUNCTION_H
