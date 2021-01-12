#include "Activation.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include <iostream>
#include <fstream>
#include <string>

    class Minerva
    {
    private:
        static void pushToFile(std::ofstream &file, void *p, size_t size);

    public:
        static void exportToFile(NeuralNetwork *network, std::string fileName);
        static NeuralNetwork *importFromFile(std::string fileName);
    };
