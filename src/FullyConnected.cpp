#include "FullyConnected.h"
FullyConnected::FullyConnected(uint16_t p_c, Activation::ActivationType act, const float lr) : Layer(act, lr)
{
    this->p_count = p_c;
}

void FullyConnected::init(uint16_t inX, uint16_t inY, uint16_t inZ)
{
    std::cout << inX << ":" << inY << ":" << inZ << std::endl;
    this->i_size = inX * inY * inZ;
    weights = new MNC::Matrix(p_count, i_size);
    weights->randomize();
    bias = new MNC::Matrix(p_count, 1);
    bias->fill(0.5);
    out = new MNC::Matrix(p_count, 1);
    outDer = new MNC::Matrix(p_count, 1);
}

MNC::Matrix FullyConnected::feed_forward(MNC::Matrix &in)
{
    MNC::Matrix r = *weights * in;
    r += *bias;
    *outDer = r;
    Activation::derivative(*outDer, activationType);
    Activation::activate(r, activationType);
    *out = r;
    return r;
}

MNC::Matrix FullyConnected::back_propagation(const MNC::Matrix &in, const MNC::Matrix &inDer, const MNC::Matrix &err)
{
    MNC::Matrix gradient(err.rows, err.columns);
    gradient = err;
    gradient *= learning_rate;
    MNC::Matrix delta = gradient * in.transpose();
    MNC::Matrix ret = weights->transpose() * err;
    ret.hadamard(inDer);
    *weights -= delta;
    *bias -= gradient;
    return ret;
}

void FullyConnected::getOutDimensions(uint16_t &outX, uint16_t &outY, uint16_t &outZ)
{
    outX = 1;
    outY = p_count;
    outZ = 1;
}