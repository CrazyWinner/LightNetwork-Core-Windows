#pragma once
#include "Matrix.h"
#include "Activation.h"

class Layer
{
public:
    float learning_rate;
    Activation::ActivationType activationType;
    MNC::Matrix *weights;
    MNC::Matrix *bias;
    MNC::Matrix *out;
    MNC::Matrix *outDer;
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
    virtual void init(uint16_t inX, uint16_t inY, uint16_t inZ) = 0;
    virtual void getOutDimensions(uint16_t &outX, uint16_t &outY, uint16_t &outZ) = 0;
};
