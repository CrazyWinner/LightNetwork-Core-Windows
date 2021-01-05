#include "Layer.h"

using namespace LightNetwork;

Layer::Layer(int i_s, int p_c, Activation *act, const float lr)
{
    activator = act;
    this->i_size = i_s;
    this->p_count = p_c;
    weights = new Matrix(p_count, i_size);
    weights->randomize();
    bias = new Matrix(p_count, 1);
    bias->fill();
    out = new Matrix(p_count, 1);
    outDer = new Matrix(p_count, 1);
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

Matrix Layer::feed_forward(Matrix &in)
{
    Matrix r = *weights * in;
    r += *bias;
    *outDer = r;
    activator->derivative(*outDer);
    activator->activate(r);
    *out = r;
    return r;
}

void Layer::back_propagation(Matrix &in, Matrix &inDer, Matrix &err)
{
    Matrix gradient(err.rows,err.columns);
    gradient = err;
    gradient *= learning_rate;
    Matrix delta = gradient * in.transpose();
    err = weights->transpose() * err;
    err.hadamard(inDer);
    *weights -= delta;
    *bias -= gradient;
}
