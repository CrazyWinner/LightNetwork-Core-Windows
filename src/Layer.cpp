#include "Layer.h"

Layer::Layer(uint16_t i_s, uint16_t p_c, Activation *act, const float lr)
{
    activator = act;
    this->i_size = i_s;
    this->p_count = p_c;
    weights = new MNC::Matrix(p_count, i_size);
    weights->randomize();
    bias = new MNC::Matrix(p_count, 1);
    bias->fill();
    out = new MNC::Matrix(p_count, 1);
    outDer = new MNC::Matrix(p_count, 1);
    this->learning_rate = lr;
}

Layer::~Layer()
{
    delete activator;
    delete out;
    delete outDer;
    delete weights;
    delete bias;
}

MNC::Matrix Layer::feed_forward(MNC::Matrix &in)
{
    MNC::Matrix r = *weights * in;
    r += *bias;
    *outDer = r;
    activator->derivative(*outDer);
    activator->activate(r);
    *out = r;
    return r;
}

void Layer::back_propagation(MNC::Matrix &in, MNC::Matrix &inDer, MNC::Matrix &err)
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
