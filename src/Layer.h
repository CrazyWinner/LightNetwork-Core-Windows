#pragma once
#include "Matrix.h"
#include "Activation.h"
#include <fstream>
class Layer
{
public:
    float learning_rate;
    enum LayerType : uint8_t {FULLY_CONNECTED = 0, CONVOLUTIONAL, MAX_POOLING};
    Activation::ActivationType activationType;
    MNC::Matrix *weights = nullptr;
    MNC::Matrix *bias = nullptr;
    MNC::Matrix *out = nullptr;
    MNC::Matrix *outDer = nullptr;
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
    virtual MNC::Matrix feed_forward(MNC::Matrix &in) = 0;
    virtual MNC::Matrix back_propagation(const MNC::Matrix &in, const MNC::Matrix &inDer, const MNC::Matrix &err) = 0;
    virtual void init(uint32_t inX, uint32_t inY, uint32_t inZ) = 0;
    virtual void getOutDimensions(uint32_t &outX, uint32_t &outY, uint32_t &outZ) = 0;
    virtual void save(std::ofstream* file) = 0;
    virtual void load(std::ifstream* file, uint32_t inX, uint32_t inY, uint32_t inZ) = 0;
};
