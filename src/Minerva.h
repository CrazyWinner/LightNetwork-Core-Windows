#include "Activation.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include <iostream>
#include <fstream>
#include <string>
#include "FullyConnected.h"
#include "Conv2D.h"
#include "MaxPooling.h"
class Minerva
{
private:
    static void pushToFile(std::ofstream &file, void *p, size_t size);

public:
    static void exportToFile(NeuralNetwork& network, std::string fileName);
    static void importFromFile(NeuralNetwork& network, const std::string& fileName);
};
