#include <iostream>
#include <time.h>
#include <string>
#include "Minerva.h"
#include "HighResClock.h"
#include "Flatten.h"
#include "MnistImporter.h"
#define LR 0.002
NeuralNetwork nn;
uint32_t guesses, correctGuesses;
bool isTraining = false;
int trainIndex = 0;
int mostAccurate = 90;

/*
This example will teach 14x14 mnist characters
*/
int getMaxVal(Matrix3D &x);
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
	//	nn.addLayer(new Conv2D(3, 3, 2, 1, 1, Activation::RELU, LR));
	//	nn.addLayer(new MaxPooling(2,2));
		nn.addLayer(new Conv2D(3, 3, 2, 0, 0, Activation::RELU, LR));
		nn.addLayer(new MaxPooling(2,2));
	    nn.addLayer(new Flatten());
		nn.addLayer(new FullyConnected(25, Activation::RELU, LR));
		nn.addLayer(new FullyConnected(25, Activation::RELU, LR));
		nn.addLayer(new FullyConnected(10, Activation::SIGMOID, LR));
	}
	else
	{
		Minerva::importFromFile(nn, "deneme");
		float deneme[196];
		for(int bb = 0; bb < 196; bb++){
			deneme[bb] = (float)bb * 0.01; 
		}
		Matrix3D denemeMatrix = Matrix3D::fromArray(14,14,1,deneme);
		Matrix3D sonuc = nn.guess(denemeMatrix);
		denemeMatrix.printDebug(0);
		sonuc.printDebug(0);
		return 0;
		//isTraining = true;
		//return 0;
		
	}


	std::cout << "Layer count:" << nn.layers.size() << std::endl;
	Timer t(true, Timer::MILLISECONDS);
	while (true)
	{
		trainIndex = rand() % 60000;
		Matrix3D in = m.getInAt(trainIndex);
		Matrix3D out = m.getOutAt(trainIndex);
		Matrix3D guessed = nn.guess(in);
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
			if ((correctGuesses / 10) > mostAccurate && isTraining)
			{		
				Minerva::exportToFile(nn, "deneme");
				std::cout << "Saved" << std::endl;
				mostAccurate = (correctGuesses / 10);
			}
			correctGuesses = 0;
			t.printElapsed("train");
		}
	}

	return 0;
}

int getMaxVal(Matrix3D &x)
{
	float maxVal = x.at(0, 0, 0);
	float maxId = 0;
	for (uint32_t i = 0; i < x.sizeY; i++)
	{
		if (x.at(0, i, 0) > maxVal)
		{
			maxVal = x.at(0, i, 0);
			maxId = i;
		}
	}
	return maxId;
}