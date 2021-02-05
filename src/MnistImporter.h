#pragma once
#include "Matrix3D.h"
#include <iostream>
#include <fstream>
class MnistImporter
{
private:
    uint16_t resDivider = 2;
    std::ifstream *imFile;
    std::ifstream *laFile;
    uint32_t rows, cols;

public:
    MnistImporter(const char *imagesFile, const char *labelsFile);
    Matrix3D getInAt(uint32_t id);
    Matrix3D getOutAt(uint32_t id);
    void readMsbFirst(std::ifstream* &file, void *ptr, size_t size);
    ~MnistImporter();
};