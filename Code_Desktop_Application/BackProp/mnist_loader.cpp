#include "pch.h"
#include "mnist_loader.h"
#include <fstream>
#include <stdexcept>

std::vector<MnistImage> load_mnist_batch(const std::string& filename) {
	std::vector<MnistImage> batch;
	std::ifstream file(filename, std::ios::binary);
	if (!file) throw std::runtime_error("Failed to open MNIST batch file");

	while (true) {
		MnistImage img;
		file.read(reinterpret_cast<char*>(img.data), sizeof(float) * 784);
		if (!file) break; // EOF or partial read

		file.read(reinterpret_cast<char*>(&img.label), sizeof(uint8_t));
		if (!file) break;

		batch.push_back(img);
	}

	return batch;
}
