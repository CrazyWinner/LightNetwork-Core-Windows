
#include "Matrix.h"
#include <iostream>
#include <fstream>
class MnistImporter
{
private:
    int resDivider = 2;
    std::ifstream* imFile;
    std::ifstream* laFile;
    uint16_t rows, cols;
public:
    MnistImporter(const char* imagesFile, const char* labelsFile);
    MNC::Matrix getInAt(uint32_t id);
    MNC::Matrix getOutAt(uint32_t id);
    void readMsbFirst(std::ifstream& file,char* ptr, size_t size);
    ~MnistImporter();
};