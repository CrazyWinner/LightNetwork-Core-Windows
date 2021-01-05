#include "ActivationFunctions.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include <iostream>
#include <fstream>
#include <string>
namespace LightNetwork
{
    class LightNetworkHelper
    {
    public:
        static void exportToFile(NeuralNetwork *network, std::string fileName);
        static NeuralNetwork *importFromFile(char *fileName);
    };
} // namespace LightNetwork