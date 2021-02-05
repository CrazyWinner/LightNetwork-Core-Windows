#include "FullyConnected.h"
FullyConnected::FullyConnected(uint32_t p_c, Activation::ActivationType act, const float lr) : Layer(act, lr)
{
    p_count = p_c;
}

void FullyConnected::init(uint32_t inX, uint32_t inY, uint32_t inZ)
{
    i_size = inX * inY * inZ;
    weights = new Matrix3D(i_size, p_count, 1);
    weights->randomize();
    bias = new Matrix3D(1, p_count, 1);
    bias->fill(0.5);
    out = new Matrix3D(1, p_count, 1);
    outDer = new Matrix3D(1, p_count, 1);
}

Matrix3D FullyConnected::feed_forward(Matrix3D &in)
{
    Matrix3D r = *weights * in;
    r += *bias;
    *outDer = r;
    Activation::derivative(*outDer, activationType);
    Activation::activate(r, activationType);
    *out = r;
    return r;
}

Matrix3D FullyConnected::back_propagation(const Matrix3D &in, const Matrix3D &err)
{
    Matrix3D gradient(err.sizeX, err.sizeY, 1);
    gradient = err;
    gradient *= learning_rate;
    Matrix3D delta = gradient * in.transpose();
    Matrix3D ret = weights->transpose() * err;
    *weights -= delta;
    *bias -= gradient;
    return ret;
}

void FullyConnected::getOutDimensions(uint32_t &outX, uint32_t &outY, uint32_t &outZ)
{
    outX = 1;
    outY = p_count;
    outZ = 1;
}

void FullyConnected::save(std::ofstream* file){
    uint8_t type = FULLY_CONNECTED;
    file->write((char*)&type, sizeof(type));
    file->write((char*)&activationType,sizeof(activationType));
    file->write((char*)&learning_rate, sizeof(learning_rate));
    file->write((char*)&p_count, sizeof(p_count));
    file->write((char*)weights->data, sizeof(float) * weights->sizeX * weights->sizeY * weights->sizeZ);
    file->write((char*)bias->data, sizeof(float) * bias->sizeX * bias->sizeY * bias->sizeZ);
}

void FullyConnected::load(std::ifstream* file, uint32_t inX, uint32_t inY, uint32_t inZ)
{
    
    i_size = inX * inY * inZ;
    weights = new Matrix3D(i_size, p_count, 1);
    file->read((char*)weights->data, sizeof(float) * weights->sizeX * weights->sizeY * weights->sizeZ);
    bias = new Matrix3D(1, p_count, 1);
    file->read((char*)bias->data, sizeof(float) * bias->sizeX * bias->sizeY * bias->sizeZ);
    out = new Matrix3D(1, p_count, 1);;
    outDer = new Matrix3D(1, p_count, 1);
}
