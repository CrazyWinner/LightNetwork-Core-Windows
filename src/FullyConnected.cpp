#include "FullyConnected.h"
FullyConnected::FullyConnected(uint32_t p_c, Activation::ActivationType act, const float lr) : Layer(act, lr)
{
    this->p_count = p_c;
}

void FullyConnected::init(uint32_t inX, uint32_t inY, uint32_t inZ)
{
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
    file->write((char*)weights->data, sizeof(float) * weights->columns * weights->rows);
    file->write((char*)bias->data, sizeof(float) * bias->columns * bias->rows);
}

void FullyConnected::load(std::ifstream* file, uint32_t inX, uint32_t inY, uint32_t inZ)
{
    
    this->i_size = inX * inY * inZ;
    weights = new MNC::Matrix(p_count, i_size);
    file->read((char*)weights->data, sizeof(float) * weights->columns * weights->rows);
    bias = new MNC::Matrix(p_count, 1);
    file->read((char*)bias->data, sizeof(float) * bias->columns * bias->rows);
    out = new MNC::Matrix(p_count, 1);
    outDer = new MNC::Matrix(p_count, 1);   
}
