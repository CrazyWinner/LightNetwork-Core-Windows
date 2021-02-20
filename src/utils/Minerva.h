#include "NeuralNetwork.h"
#include "Matrix3D.h"
#include <iostream>
#include <fstream>
#include <string>
#include "../layers/Activation.h"
#include "../layers/FullyConnected.h"
#include "../layers/Conv2D.h"
#include "../layers/MaxPooling.h"
#include "../layers/Flatten.h"
class Minerva
{
private:
    static void pushToFile(std::ofstream &file, void *p, size_t size);

public:
    static void exportToFile(NeuralNetwork& network, const std::string& fileName);
    static void importFromFile(NeuralNetwork& network, const std::string& fileName);
};
