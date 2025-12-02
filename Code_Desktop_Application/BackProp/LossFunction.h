#pragma once

#include <cmath>

/**
 * @author David Muttenthaler
 * @date 23-06-2025
 *
 * @brief Abstrakte Basisklasse für Verlustfunktionen in neuronalen Netzwerken.
 *
 * Diese Klasse definiert ein generisches Interface für Verlustfunktionen,
 * die zum Trainieren von Modellen verwendet werden. Die Methoden `compute` und `derivative`
 * müssen von abgeleiteten Klassen implementiert werden.
 *
 * @tparam T Der verwendete Datentyp (z. B. float oder double)
 */
template<typename T>
class LossFunction {
public:
	virtual ~LossFunction() = default;

	/**
	 * @brief Berechnet den Verlustwert zwischen Vorhersage und Zielwert.
	 *
	 * @param predicted Zeiger auf Array mit Vorhersagewerten (z. B. Ausgaben des Modells)
	 * @param target Zeiger auf Array mit Sollwerten (Ground Truth)
	 * @param size Anzahl der Elemente in den Arrays
	 * @return T Der berechnete Verlustwert (Loss)
	 */
	virtual T compute(const T* predicted, const T* target, std::size_t size) const = 0;

	/**
	 * @brief Berechnet den Gradienten des Verlusts bezüglich der Vorhersagewerte.
	 *
	 * @param predicted Zeiger auf Array mit Vorhersagewerten
	 * @param target Zeiger auf Array mit Sollwerten
	 * @param output_grad Zeiger auf Array, in das die Gradienten geschrieben werden
	 * @param size Anzahl der Elemente in den Arrays
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
 * @tparam T Numeric data type (e.g., float or double).
 */
template<typename T>
class MSELoss : public LossFunction<T> {
public:
	T compute(const T* predicted, const T* target, std::size_t size) const override {
		T sum = 0;
		for (std::size_t i = 0; i < size; ++i) {
			T diff = predicted[i] - target[i];
			sum += diff * diff;
		}
		return sum / static_cast<T>(size);
	}

	void derivative(const T* predicted, const T* target, T* output_grad, std::size_t size) const override {
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
 * @tparam T Numeric data type (e.g., float or double).
 */
template<typename T>
class CrossEntropyLoss : public LossFunction<T> {
public:
	T compute(const T* predicted, const T* target, std::size_t size) const override {
		T loss = 0;
		for (std::size_t i = 0; i < size; ++i) {
	
			T p = std::max(predicted[i], static_cast<T>(1e-12));
			loss -= target[i] * std::log(p);
		}
		return loss;
	}

	void derivative(const T* predicted, const T* target, T* output_grad, std::size_t size) const override {
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
 * @tparam T Numeric data type (e.g., float or double).
 */
template<typename T>
class SoftmaxCrossEntropyLoss : public LossFunction<T> {
public:
	T compute(const T* logits, const T* target, std::size_t size) const override {
		
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

	void derivative(const T* logits, const T* target, T* output_grad, std::size_t size) const override {
	
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
 * @tparam T Numeric data type (e.g., float or double).
 */
template<typename T>
class SigmoidBinaryCrossEntropyLoss : public LossFunction<T> {
public:
	T compute(const T* logits, const T* targets, std::size_t size) const override {
		T loss = 0;
		for (std::size_t i = 0; i < size; ++i) {
			T z = logits[i];
			T y = targets[i];

			if (z >= 0) {
				loss += (std::log(1 + std::exp(-z)) + (1 - y) * z);
			}
			else {
				loss += (std::log(1 + std::exp(z)) - y * z);
			}
		}
		return loss / static_cast<T>(size);
	}

	void derivative(const T* logits, const T* targets, T* output_grad, std::size_t size) const override {
		for (std::size_t i = 0; i < size; ++i) {
			T z = logits[i];
			T y = targets[i];

			T sigmoid = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-z));
			output_grad[i] = (sigmoid - y) / static_cast<T>(size);
		}
	}
};


//ChatGPT Loss Funktions: wurden von ChatGPT geschrieben und nicht getestet
//template<typename T>
//class HuberLoss : public LossFunction<T> {
//public:
//	explicit HuberLoss(T delta = static_cast<T>(1)) : delta_(delta) {}
//
//	T compute(const T* predicted, const T* target, std::size_t size) const override {
//		T loss = 0;
//		for (std::size_t i = 0; i < size; ++i) {
//			T diff = predicted[i] - target[i];
//			T abs_diff = std::abs(diff);
//			if (abs_diff <= delta_) {
//				loss += 0.5 * diff * diff;
//			}
//			else {
//				loss += delta_ * (abs_diff - 0.5 * delta_);
//			}
//		}
//		return loss / static_cast<T>(size);
//	}
//
//	void derivative(const T* predicted, const T* target, T* output_grad, std::size_t size) const override {
//		for (std::size_t i = 0; i < size; ++i) {
//			T diff = predicted[i] - target[i];
//			if (std::abs(diff) <= delta_) {
//				output_grad[i] = diff / static_cast<T>(size);
//			}
//			else {
//				output_grad[i] = delta_ * ((diff > 0) ? 1 : -1) / static_cast<T>(size);
//			}
//		}
//	}
//private:
//	T delta_;
//};
//
//
//template<typename T>
//class KLDivergenceLoss : public LossFunction<T> {
//public:
//	T compute(const T* predicted, const T* target, std::size_t size) const override {
//		T loss = 0;
//		for (std::size_t i = 0; i < size; ++i) {
//			T p = std::max(predicted[i], static_cast<T>(1e-12));
//			T q = std::max(target[i], static_cast<T>(1e-12));
//			loss += q * (std::log(q) - std::log(p));
//		}
//		return loss;
//	}
//
//	void derivative(const T* predicted, const T* target, T* output_grad, std::size_t size) const override {
//		for (std::size_t i = 0; i < size; ++i) {
//			T p = std::max(predicted[i], static_cast<T>(1e-12));
//			T q = std::max(target[i], static_cast<T>(1e-12));
//			output_grad[i] = -q / p;
//		}
//	}
//};
//
//
//template<typename T>
//class MAELoss : public LossFunction<T> {
//public:
//	T compute(const T* predicted, const T* target, std::size_t size) const override {
//		T loss = 0;
//		for (std::size_t i = 0; i < size; ++i) {
//			loss += std::abs(predicted[i] - target[i]);
//		}
//		return loss / static_cast<T>(size);
//	}
//
//	void derivative(const T* predicted, const T* target, T* output_grad, std::size_t size) const override {
//		for (std::size_t i = 0; i < size; ++i) {
//			T diff = predicted[i] - target[i];
//			output_grad[i] = ((diff > 0) ? 1 : (diff < 0 ? -1 : 0)) / static_cast<T>(size);
//		}
//	}
//};

