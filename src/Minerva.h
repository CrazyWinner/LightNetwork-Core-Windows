#include "Activation.h"
#include "NeuralNetwork.h"
#include "Matrix3D.h"
#include <iostream>
#include <fstream>
#include <string>
#include "FullyConnected.h"
#include "Conv2D.h"
#include "MaxPooling.h"
#include "Flatten.h"
class Minerva
{
private:
    static void pushToFile(std::ofstream &file, void *p, size_t size);

public:
    static void exportToFile(NeuralNetwork& network, const std::string& fileName);
    static void importFromFile(NeuralNetwork& network, const std::string& fileName);
};
