#include <iostream>
#include <vector>
#include "math.h"
#include <time.h>
#include <chrono>
#include "Minerva.h"
#include <string>
#include "HighResClock.h"
#include <unistd.h>

NeuralNetwork nn(196); // Create a neural network with 2 input neurons, input neurons will be created automatically
uint16_t guesses, correctGuesses;
bool isTraining = true;
int main()
{

	/*
    Add 2 hidden layers and 1 output layer. 
    You don't need to specify output layer. Last layer will be output layer automatically.
   */

	srand((unsigned)time(NULL));
	nn.addLayer(16, new SIGMOID(), 0.01);
	nn.addLayer(16, new SIGMOID(), 0.01);
	nn.addLayer(10, new SIGMOID(), 0.01);
	if (!isTraining)
		nn = *Minerva::importFromFile((std::string) "a");
	std::cout << "Size:" << nn.layers.size() << std::endl;
	Timer t(true, Timer::MILLISECONDS);
	while (true)
	{

		/*

        This example will teach a circle in data. We will pick a random coordinate between (0,0) and (1,1)
		and calculate it's distance to middle. If it's in circle nn should output 1 otherwise 0.
       */

		float a = ((double)rand() / (RAND_MAX + 1.0));
		float b = ((double)rand() / (RAND_MAX + 1.0));

		float toMiddle = std::sqrt((a - 0.5) * (a - 0.5) + (b - 0.5) * (b - 0.5));

		float input[196] = {a, b};

		MNC::Matrix inputMatrix = MNC::Matrix::fromArray(196, 1, input);
		float guessed = nn.guess(inputMatrix).at(0, 0);
		guessed = guessed < 0.5 ? 0 : 1;
		float region = (toMiddle < 0.25) ? 0 : 1;

		if (guessed == region)
		{
			correctGuesses++;
		}
		guesses++;
		float rrrr[10] = {region};
		MNC::Matrix expectedOutputMatrix = MNC::Matrix::fromArray(10, 1, rrrr);

		if (isTraining)
			nn.train(inputMatrix, expectedOutputMatrix);
		if (guesses == 10000)
		{
			std::cout << "Dogruluk orani %" << correctGuesses / 100 << std::endl;
			guesses = 0;
			if ((correctGuesses / 100) > 94 && isTraining)
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