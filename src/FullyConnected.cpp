#include "FullyConnected.h"
FullyConnected::FullyConnected(uint16_t p_c, Activation::ActivationType act, const float lr) : Layer(act,lr)
{
    this->p_count = p_c;
}

void FullyConnected::init(uint16_t inX, uint16_t inY, uint16_t inZ){
this->i_size = inX * inY * inZ;
weights = new MNC::Matrix(p_count, i_size);
weights->randomize();
bias = new MNC::Matrix(p_count, 1);
bias->fill();
out = new MNC::Matrix(p_count, 1);
outDer = new MNC::Matrix(p_count, 1);
}


MNC::Matrix FullyConnected::feed_forward(MNC::Matrix &in)
{
    MNC::Matrix r = *weights * in;
    r += *bias;
    *outDer = r;
    Activation::derivative(*outDer, activationType);
    Activation::activate(r,activationType);
    *out = r;
    return r;
}

void FullyConnected::back_propagation(MNC::Matrix &in, MNC::Matrix &inDer, MNC::Matrix &err)
{
    MNC::Matrix gradient(err.rows, err.columns);
    gradient = err;
    gradient *= learning_rate;
    MNC::Matrix delta = gradient * in.transpose();
    err = weights->transpose() * err;
    err.hadamard(inDer);
    *weights -= delta;
    *bias -= gradient;
}

void FullyConnected::getOutDimensions(uint16_t& outX, uint16_t& outY, uint16_t& outZ){
   outX = 1;
   outY = p_count;
   outZ = 1;

}