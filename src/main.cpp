#include <iostream>
#include <time.h>
#include <string>
#include "Minerva.h"
#include "HighResClock.h"
#include "MnistImporter.h"
#define LR 0.01
NeuralNetwork nn;
uint32_t guesses, correctGuesses;
bool isTraining = true;
int trainIndex = 0;

/*
This example will teach 14x14 mnist characters
*/
int getMaxVal(MNC::Matrix &x);
int main()
{
	MnistImporter m("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	/*
    Add 2 hidden layers and 1 output layer. 
    You don't need to specify output layer. Last layer will be the output layer.
   */

	srand((unsigned)time(NULL));
	if (isTraining)
	{
		nn.init(14,14,1);
		nn.addLayer(new Conv2D(3, 3, 5, 1, 1, Activation::LEAKY_RELU, LR));
		nn.addLayer(new MaxPooling(2,2));
		nn.addLayer(new Conv2D(3, 5, 5, 1, 1, Activation::LEAKY_RELU, LR));
		nn.addLayer(new MaxPooling(2,2));
	    nn.addLayer(new FullyConnected(16, Activation::LEAKY_RELU, LR));
		nn.addLayer(new FullyConnected(16, Activation::LEAKY_RELU, LR));
		nn.addLayer(new FullyConnected(10, Activation::SIGMOID, LR));
	}
	else
	{
		Minerva::importFromFile(nn, "deneme");
	}

	std::cout << "Layer count:" << nn.layers.size() << std::endl;
	Timer t(true, Timer::MILLISECONDS);
	while (true)
	{
		trainIndex = rand() % 60000;
		MNC::Matrix in = m.getInAt(trainIndex);
		MNC::Matrix out = m.getOutAt(trainIndex);
		MNC::Matrix guessed = nn.guess(in);
		int desired = getMaxVal(out);
		int result = getMaxVal(guessed);
		if (desired == result)
		{
			correctGuesses++;
		}
		guesses++;
		if (isTraining)
			nn.train(in, out);
		if (guesses == 1000)
		{
			std::cout << (isTraining ? "Train" : "Test") << " accuracy %" << correctGuesses / 10 << std::endl;
			guesses = 0;
			if ((correctGuesses / 10) > 90 && isTraining)
			{
				std::cout << "Saved" << std::endl;
				Minerva::exportToFile(nn, (std::string) "deneme");
				return 0;
			}
			correctGuesses = 0;
			t.printElapsed("train");
		}
	}

	return 0;
}

int getMaxVal(MNC::Matrix &x)
{
	float maxVal = x.at(0, 0);
	float maxId = 0;
	for (uint32_t i = 0; i < x.rows; i++)
	{
		if (x.at(i, 0) > maxVal)
		{
			maxVal = x.at(i, 0);
			maxId = i;
		}
	}
	return maxId;
}