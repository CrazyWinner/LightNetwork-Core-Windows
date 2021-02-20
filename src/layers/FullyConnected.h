#pragma once
#include "Layer.h"
class FullyConnected : public Layer
{
private:
    uint32_t i_size, p_count;

public:
    FullyConnected(uint32_t p_c, Activation::ActivationType act, const float lr);
    Matrix3D feed_forward(Matrix3D &in);
    Matrix3D back_propagation(Matrix3D &in, Matrix3D &err);
    void init(uint32_t inX, uint32_t inY, uint32_t inZ);
    void getOutDimensions(uint32_t &outX, uint32_t &outY, uint32_t &outZ);
    void save(std::ofstream* file);
    void load(std::ifstream* file, uint32_t inX, uint32_t inY, uint32_t inZ);
};
