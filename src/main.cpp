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
NeuralNetwork nn(196); // Create a neural network with 2 input neurons, input neurons will be created automatically
uint16_t guesses, correctGuesses;
bool isTraining = true;
int trainIndex = 0;
/*
This example will teach 14x14 mnist characters

*/
int getMaxVal(MNC::Matrix& x);
int main()
{ 
	MnistImporter m((char*)"train-images.idx3-ubyte",(char*)"train-labels.idx1-ubyte");
	/*
    Add 2 hidden layers and 1 output layer. 
    You don't need to specify output layer. Last layer will be output layer automatically.
   */

	srand((unsigned)time(NULL));
	nn.addLayer(16, Activation::SIGMOID, 0.05);

	nn.addLayer(16, Activation::SIGMOID, 0.05);
	nn.addLayer(10, Activation::SIGMOID, 0.05);
	if (!isTraining)
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
			if ((correctGuesses / 10) > 96 && isTraining)
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
    for(int i = 0; i < x.rows; i++){
		if(x.at(i,0) > maxVal){
			maxVal = x.at(i,0);
			maxId = i;
		}
	}
	return maxId;
}