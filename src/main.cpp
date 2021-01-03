#include <iostream>
#include "Matrix.h"
#include <vector>
#include "math.h"
#include "NeuralNetwork.h"
#include <time.h>
#include <unistd.h>
#include "ActivationFunctions.h"
#include <chrono>
#include <iomanip>      // std::setprecision
#define LN LightNetwork
LN::NeuralNetwork nn(2);
int tahminler, dogruTahminler;
int main()
{

   
	srand((unsigned)time(NULL));
	nn.addLayer(3, new LN::SIGMOID());
	nn.addLayer(1, new LN::SIGMOID());
    float arr[] = {0,0};
	while (true)
	{ 
        float a = ((double)rand() / (RAND_MAX + 1.0));
		float b = ((double)rand() / (RAND_MAX + 1.0));
		float toMiddle = std::sqrt((a - 0.5) * (a - 0.5) + (b - 0.5) * (b - 0.5));
		arr[0] = a;
		arr[1] = b;
		LN::Matrix nnm = LN::Matrix::fromArray(2, 1, arr);
		float guessed = nn.guess(nnm).at(0, 0);
		guessed = guessed < 0.5 ? 0 : 1;
		float region = (toMiddle < 0.25) ? 0 : 1;
		if (guessed == region)
		{
			dogruTahminler++;
		}
		tahminler++;
		LN::Matrix aa = LN::Matrix::fromArray(2, 1, arr);
		LN::Matrix bb = LN::Matrix::fromArray(1, 1, &region);
		nn.train(aa, bb);
		
		if (tahminler == 10000)
		{
			std::cout << "Dogruluk orani %" << dogruTahminler / 100 << std::endl;
			tahminler = 0;
			dogruTahminler = 0;
		}
	}
	return 0;
}