#pragma once
#include "Layer.h"
class Conv2D : public Layer
{
private:
    uint32_t filter_size_X, filter_size_Y, filter_count, padding_X, padding_Y;
    uint32_t i_X, i_Y, i_Z;

public:
    Conv2D(uint32_t fsx, uint32_t fsy, uint32_t f_c, uint32_t pd_x, uint32_t pd_y, Activation::ActivationType act, const float lr);
    Matrix3D feed_forward(Matrix3D &in);
    Matrix3D back_propagation(Matrix3D &in, Matrix3D &err);
    void init(uint32_t inX, uint32_t inY, uint32_t inZ);
    void getOutDimensions(uint32_t &outX, uint32_t &outY, uint32_t &outZ);
    void save(std::ofstream* file);
    void load(std::ifstream* file, uint32_t inX, uint32_t inY, uint32_t inZ);
};
