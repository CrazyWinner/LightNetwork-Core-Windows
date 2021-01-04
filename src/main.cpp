#include <iostream>
#include <vector>
#include "math.h"
#include <time.h>
#include <chrono>
#include "LightNetwork.h"
 

#define LN LightNetwork
LN::NeuralNetwork nn(2); // Create a neural network with 2 input neurons, input neurons will be created automatically
int guesses, correctGuesses;

int main()
{

   /*
    Add 2 hidden layers and 1 output layer. 
    You don't need to specify output layer. Last layer will be output layer automatically.
   */
   
   	srand((unsigned)time(NULL));
	   
	nn.addLayer(16, new LN::RELU(), 0.1);
	nn.addLayer(16, new LN::RELU(), 0.1);
	nn.addLayer(1, new LN::SIGMOID(), 0.1);
	while (true)
	{   

		/*

        This example will teach a circle in data. We will pick a random coordinate between (0,0) and (1,1)
		and calculate it's distance to middle. If it's in circle nn should output 1 otherwise 0.
       */
        
        float a = ((double)rand() / (RAND_MAX + 1.0));
		float b = ((double)rand() / (RAND_MAX + 1.0));
		
		float toMiddle = std::sqrt((a - 0.5) * (a - 0.5) + (b - 0.5) * (b - 0.5));
		float input[] =  {a,b};
		LN::Matrix inputMatrix = LN::Matrix::fromArray(2, 1, input);
		float guessed = nn.guess(inputMatrix).at(0, 0);
		guessed = guessed < 0.5 ? 0 : 1;
		float region = (toMiddle < 0.25) ? 0 : 1;
		if (guessed == region)
		{
			correctGuesses++;
		}
		guesses++;
		LN::Matrix expectedOutputMatrix = LN::Matrix::fromArray(1, 1, &region);
		nn.train(inputMatrix, expectedOutputMatrix);
		if (guesses == 10000)
		{
			std::cout << "Dogruluk orani %" << correctGuesses / 100 << std::endl;
			guesses = 0;
			correctGuesses = 0;
		}
	}
	
	return 0;
}