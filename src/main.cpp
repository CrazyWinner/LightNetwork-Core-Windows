#include <iostream>
#include <vector>
#include "math.h"
#include <time.h>
#include <chrono>
#include "Minerva.h"
#include <string>
#include "HighResClock.h"
#include <unistd.h>
#include "MnistImporter.h"
#include "FullyConnected.h"
#include "Conv2D.h"
#include "MaxPooling.h"
NeuralNetwork nn(14,14,1); 
uint32_t guesses, correctGuesses;
bool isTraining = true;
int trainIndex = 0;

/*
This example will teach 14x14 mnist characters
*/
int getMaxVal(MNC::Matrix& x);
int main()
{ 
	MnistImporter m("train-images.idx3-ubyte","train-labels.idx1-ubyte");
	/*
    Add 2 hidden layers and 1 output layer. 
    You don't need to specify output layer. Last layer will be the output layer.
   */

	srand((unsigned)time(NULL));
	nn.addLayer(new Conv2D(3,5,1,Activation::LEAKY_RELU,0.01)); 
	nn.addLayer(new MaxPooling(2)); 
	nn.addLayer(new Conv2D(3,10,1,Activation::LEAKY_RELU,0.01)); 
	nn.addLayer(new MaxPooling(2)); 
	nn.addLayer(new FullyConnected(30, Activation::LEAKY_RELU, 0.01));
    nn.addLayer(new FullyConnected(30, Activation::LEAKY_RELU, 0.01)); 
    nn.addLayer(new FullyConnected(10, Activation::SIGMOID, 0.01));


	
	if (!isTraining && false)
		nn = *Minerva::importFromFile((std::string) "a");
		
	std::cout << "Size:" << nn.layers.size() << std::endl;
	Timer t(true, Timer::MILLISECONDS);
	while (true)
	{
		trainIndex = rand() % 60000;
		MNC::Matrix in = m.getInAt(trainIndex);
		MNC::Matrix out = m.getOutAt(trainIndex);
		MNC::Matrix guessed = nn.guess(in);
        int desired = getMaxVal(out);
		int result = getMaxVal(guessed);
		if(desired == result){
			correctGuesses++;
		}
		guesses++;
		if (isTraining)
			nn.train(in, out);
		if (guesses == 1000)
		{
			std::cout << "Dogruluk orani %" << correctGuesses / 10 << std::endl;
			guesses = 0;
			if ((correctGuesses / 10) > 96 && isTraining && false)
			{
				std::cout << "Saved" << std::endl;
				Minerva::exportToFile(&nn, (std::string) "a");
				return 0;
			}
			correctGuesses = 0;
			t.printElapsed("train");
		}
	}

	return 0;
}

int getMaxVal(MNC::Matrix& x){
    float maxVal = x.at(0,0);
	float maxId = 0;
    for(uint32_t i = 0; i < x.rows; i++){
		if(x.at(i,0) > maxVal){
			maxVal = x.at(i,0);
			maxId = i;
		}
	}
	return maxId;
}