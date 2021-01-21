#include "Conv2D.h"
Conv2D::Conv2D(uint16_t f_s, uint16_t f_c, uint16_t pd, Activation::ActivationType act, const float lr) : Layer(act, lr)
{
    filter_size = f_s;
    filter_count = f_c;
    padding = pd;
}
void Conv2D::init(uint16_t inX, uint16_t inY, uint16_t inZ)
{
    i_X = inX;
    i_Y = inY;
    i_Z = inZ;
    weights = new MNC::Matrix(filter_size * filter_size, filter_count);
    weights->randomize();
    bias = new MNC::Matrix(filter_count, 1);
    bias->fill(0.1);
    uint16_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    out = new MNC::Matrix(outX * outY * outZ, 1);
    outDer = new MNC::Matrix(outX * outY * outZ, 1);
}

void Conv2D::getOutDimensions(uint16_t &outX, uint16_t &outY, uint16_t &outZ)
{
    outX = i_X - filter_size + 1 + (2 * padding);
    outY = i_Y - filter_size + 1 + (2 * padding);
    outZ = i_Z * filter_count;
}

MNC::Matrix Conv2D::feed_forward(MNC::Matrix &in)
{
    uint16_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    MNC::Matrix r(outX * outY * outZ, 1);
    for (uint16_t f = 0; f < filter_count; f++)
    {
        for (uint16_t z = 0; z < i_Z; z++)
        {
            MNC::Matrix inSub = in.getSubMatrix(i_Y, i_X, z);
            MNC::Matrix o = inSub.convolve(weights->getSubMatrix(filter_size, filter_size, f), padding);
            o += bias->at(f,0);
            r.getSubMatrix(o.rows, o.columns, f * i_Z + z) = o;
        }
    }
    *outDer = r;
    Activation::derivative(*outDer, activationType);
    Activation::activate(r, activationType);
    *out = r;
    return r;
}

MNC::Matrix Conv2D::back_propagation(const MNC::Matrix &in, const MNC::Matrix &inDer, const MNC::Matrix &err)
{
    MNC::Matrix ret(i_X * i_Y * i_Z, 1);
    uint16_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    MNC::Matrix gradient(err.rows, err.columns);
    gradient = err;
    gradient *= learning_rate;
    for(uint16_t f = 0; f < filter_count; f++){
        for(uint16_t z = 0; z < i_Z; z++){
           MNC::Matrix inSub = in.getSubMatrix(i_X, i_Y, z);
           MNC::Matrix gradientSub = gradient.getSubMatrix(outX, outY, f * i_Z + z);
           float db = gradientSub.sum();
           bias->set(f, 0, bias->at(f, 0) - db);
           MNC::Matrix dw = inSub.convolve(gradientSub, padding);
           weights->getSubMatrix(filter_size, filter_size, f) -= dw;
           int dx_padding = (i_X - outX + filter_size - 1) / 2;
           MNC::Matrix dx = gradientSub.convolve(weights->getSubMatrix(filter_size, filter_size, f), dx_padding);
           ret.getSubMatrix(i_X, i_Y, z) += dx;
        }
    }
    ret.hadamard(inDer);
    return ret;
}
