// BackProp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include "model.h"
#include "ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include "mnist_loader.h"


int main()
{
	//load hexfile Model
	// Öffne die Datein im Binärmodus
	std::ifstream file_fixed("model.hex", std::ios::binary);
	if (!file_fixed) {
		std::cerr << "Fehler beim oeffnen der Datei." << std::endl;
		return 1;
	}

	std::ifstream file_trainable("model_trainable.hex", std::ios::binary);
	if (!file_trainable) {
		std::cerr << "Fehler beim oeffnen der Datei." << std::endl;
		return 1;
	}

	// Bestimme die Dateigröße
	file_fixed.seekg(0, std::ios::end);
	std::streamsize size_fixed = file_fixed.tellg();
	file_fixed.seekg(0, std::ios::beg);

	file_trainable.seekg(0, std::ios::end);
	std::streamsize size_trainable = file_trainable.tellg();
	file_trainable.seekg(0, std::ios::beg);

	// Alloziere einen Puffer
	char* model_fixed = new char[size_fixed];

	char* model_trainable = new char[size_trainable];

	// Lese die Datei in den Puffer
	if (!file_fixed.read(model_fixed, size_fixed)) {
		std::cerr << "Fehler beim Lesen der Datei." << std::endl;
		delete[] model_fixed;
		return 1;
	}

	if (!file_trainable.read(model_trainable, size_trainable)) {
		std::cerr << "Fehler beim Lesen der Datei." << std::endl;
		delete[] model_trainable;
		return 1;
	}

	// Get Pointer to beginning of files
	void* voidPtr_fixed = model_fixed;

	void* voidPtr_trainable = model_trainable; 

	model<float> myModel(voidPtr_fixed, voidPtr_trainable, ADAM, 0.01);

	myModel.Init();


	//Load TestData

	std::vector<MnistImage> images;

	try {
		images = load_mnist_batch("mnist_all_ordered.bin");
	}
	catch (const std::exception& e) {
		std::cerr << "Error loading MNIST batch: " << e.what() << std::endl;
		return 1;
	}

	std::vector<MnistImage> testimages;

	try {
		testimages = load_mnist_batch("mnist_test_all_ordered.bin");
	}
	catch (const std::exception& e) {
		std::cerr << "Error loading MNIST test batch: " << e.what() << std::endl;
		return 1;
	}



	const MnistImage* mnist_four_byte = &images[0];

	// Reinterpret the bit-patterns as floats
	static const float* mnist_test_data = reinterpret_cast<const float*>(mnist_four_byte->data);

	const int numOutputs = 10;
	float output[numOutputs] = { 0.0 };



	float expectedOutput[numOutputs] = { 0.0 };

	int Training_size = 60000;
	int batch_size = 64;
	int testsize = 10000;
	float lossbefore = 0.0;
	int trainingEpochs = 5;

	//Test on test batch
	for (int x = 0; x < testsize; x++) {
		mnist_four_byte = &testimages[x];
		mnist_test_data = reinterpret_cast<const float*>(mnist_four_byte->data);
		myModel.InferenceSRAM(mnist_test_data, output);

		expectedOutput[mnist_four_byte->label] = 1.0;

		for (int i = 0; i < numOutputs; i++) {
			if (expectedOutput[i] > 0.0f) {
				lossbefore -= std::log(output[i] + 1e-9f); // vermeidet log(0)
			}
		}
		expectedOutput[mnist_four_byte->label] = 0.0;
	}
	std::cout << "Loss before training: " << lossbefore / testsize << std::endl;

	for (int epoch = 0; epoch < trainingEpochs; epoch++) {

		for (int x = 0; x < Training_size / batch_size; x++) {
			myModel.initGradients();
			for (int i = 0; i < batch_size; i++) {
				// Training, select Dataset
				// Reinterpret the bit-patterns as floats
				mnist_four_byte = &images[x*batch_size + i];
				mnist_test_data = reinterpret_cast<const float*>(mnist_four_byte->data);
				expectedOutput[mnist_four_byte->label] = 1.0;
				myModel.Train(mnist_test_data, expectedOutput);
				expectedOutput[mnist_four_byte->label] = 0.0;
			}

			myModel.Update(batch_size); // führe Optimizer durchlauf aus
			myModel.deleteGradients();
			//std::cout << "Batch " << x + 1 << " completed. " << std::endl;
		}
		std::cout << "Epoch " << epoch + 1 << " completed. " << std::endl;

		//Test on test immages
		uint32_t correctpredictions = 0;
		float lossafter = 0.0;
		for (int x = 0; x < testsize; x++) {
			mnist_four_byte = &testimages[x];
			mnist_test_data = reinterpret_cast<const float*>(mnist_four_byte->data);
			myModel.InferenceSRAM(mnist_test_data, output);

			expectedOutput[mnist_four_byte->label] = 1.0;
			int maxprediction = 0;
			float max = output[0];
			for (int i = 0; i < numOutputs; i++) {
				if (expectedOutput[i] > 0.0f) {
					lossafter -= std::log(output[i] + 1e-9f); // vermeidet log(0)
				}
				if (output[i] > max) {
					max = output[i];
					maxprediction = i;
				}

			}
			if (maxprediction == mnist_four_byte->label) {
				correctpredictions++;
			}

			expectedOutput[mnist_four_byte->label] = 0.0;
		}
		float accuracy = (1.0 * correctpredictions) / testsize;
		std::cout << "Accuracy after epoch " << epoch+1 << " : " << accuracy << std::endl;
		std::cout << "Loss after epoch " << epoch+1 << " : " << lossafter / testsize << std::endl << std::endl;




	}
	

	//Test on test immages
	uint32_t correctpredictions = 0;
	float lossafter = 0.0;
	for (int x = 0; x < testsize; x++) {
		mnist_four_byte = &testimages[x];
		mnist_test_data = reinterpret_cast<const float*>(mnist_four_byte->data);
		myModel.InferenceSRAM(mnist_test_data, output);

		expectedOutput[mnist_four_byte->label] = 1.0;
		int maxprediction = 0;
		float max = output[0];
		for (int i = 0; i < numOutputs; i++) {
			if (expectedOutput[i] > 0.0f) {
				lossafter -= std::log(output[i] + 1e-9f); // vermeidet log(0)
			}
			if (output[i] > max) {
				max = output[i];
				maxprediction = i;
			}

		}
		if (maxprediction == mnist_four_byte->label) {
			correctpredictions++;
		}

		expectedOutput[mnist_four_byte->label] = 0.0;
	}
	float accuracy = (1.0 * correctpredictions) / testsize;
	std::cout << "Accuracy after training: " << accuracy << std::endl;
	

	//Output Results
	std::cout << "Loss before training: " << lossbefore/ testsize << std::endl;
	std::cout << "Loss after training: " << lossafter/ testsize << std::endl;



	// Speicher wieder freigeben
	delete[] model_fixed;
	delete[] model_trainable;

	
	//std::cout << std::endl << "Press Enter to exit!" << std::endl;
	//std::cin.get();

	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file



