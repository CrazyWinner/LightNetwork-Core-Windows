#pragma once
#include "Layer.h"
class MaxPooling : public Layer
{
private:
    uint32_t pooling_size_X;
    uint32_t pooling_size_Y;
    uint32_t i_X, i_Y, i_Z;
    uint32_t* inCoordX;
    uint32_t* inCoordY;
public:
    MaxPooling(uint32_t sizeX, uint32_t sizeY);
    Matrix3D feed_forward(Matrix3D &in);
    Matrix3D back_propagation(Matrix3D &in, Matrix3D &err);
    void init(uint32_t inX, uint32_t inY, uint32_t inZ);
    void getOutDimensions(uint32_t &outX, uint32_t &outY, uint32_t &outZ);
    ~MaxPooling();
    void save(std::ofstream* file);
    void load(std::ifstream* file, uint32_t inX, uint32_t inY, uint32_t inZ);
};
