#pragma once
#include "Matrix3D.h"
#include "Activation.h"
#include <fstream>
class Layer
{
   
public:
    Matrix3D *weights = nullptr;
    Matrix3D *bias = nullptr;
    Matrix3D *out = nullptr;
    Matrix3D *outDer = nullptr;   
    float learning_rate;
    enum LayerType : uint8_t {FULLY_CONNECTED = 0, CONVOLUTIONAL, MAX_POOLING, FLATTEN};
    Activation::ActivationType activationType;
    Layer(Activation::ActivationType act, const float lr)
    {
        activationType = act;
        learning_rate = lr;
    }
    ~Layer()
    {
        delete weights;
        delete bias;
        delete out;
        delete outDer;
    }
    virtual Matrix3D feed_forward(Matrix3D &in) = 0;
    virtual Matrix3D back_propagation(const Matrix3D &in, const Matrix3D &err) = 0;
    virtual void init(uint32_t inX, uint32_t inY, uint32_t inZ) = 0;
    virtual void getOutDimensions(uint32_t &outX, uint32_t &outY, uint32_t &outZ) = 0;
    virtual void save(std::ofstream* file) = 0;
    virtual void load(std::ifstream* file, uint32_t inX, uint32_t inY, uint32_t inZ) = 0;
};
