#include "Conv2D.h"
Conv2D::Conv2D(uint32_t f_s, uint32_t f_c, uint32_t pd, Activation::ActivationType act, const float lr) : Layer(act, lr)
{
    filter_size = f_s;
    filter_count = f_c;
    padding = pd;
}
void Conv2D::init(uint32_t inX, uint32_t inY, uint32_t inZ)
{
    i_X = inX;
    i_Y = inY;
    i_Z = inZ;
    weights = new MNC::Matrix(filter_size * filter_size, filter_count);
    weights->randomize();
    bias = new MNC::Matrix(filter_count, 1);
    bias->fill(0.1);
    uint32_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    out = new MNC::Matrix(outX * outY * outZ, 1);
    outDer = new MNC::Matrix(outX * outY * outZ, 1);
}

void Conv2D::getOutDimensions(uint32_t &outX, uint32_t &outY, uint32_t &outZ)
{
    outX = i_X - filter_size + 1 + (2 * padding);
    outY = i_Y - filter_size + 1 + (2 * padding);
    outZ = i_Z * filter_count;
}

MNC::Matrix Conv2D::feed_forward(MNC::Matrix &in)
{
    uint32_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    MNC::Matrix r(outX * outY * outZ, 1);
    for (uint32_t f = 0; f < filter_count; f++)
    {
        for (uint32_t z = 0; z < i_Z; z++)
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
    uint32_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    MNC::Matrix gradient(err.rows, err.columns);
    gradient = err;
    gradient *= learning_rate;
    for(uint32_t f = 0; f < filter_count; f++){
        for(uint32_t z = 0; z < i_Z; z++){
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

void Conv2D::save(std::ofstream* file){
    uint8_t type = CONVOLUTIONAL;
    file->write((char*)&type, sizeof(type));
    file->write((char*)&activationType,sizeof(activationType));
    file->write((char*)&learning_rate, sizeof(learning_rate));
    file->write((char*)&filter_size, sizeof(filter_size));
    file->write((char*)&filter_count, sizeof(filter_count));
    file->write((char*)&padding, sizeof(padding));
    file->write((char*)weights->data, sizeof(float) * weights->columns * weights->rows);
    file->write((char*)bias->data, sizeof(float) * bias->columns * bias->rows);
}

void Conv2D::load(std::ifstream* file, uint32_t inX, uint32_t inY, uint32_t inZ)
{
    i_X = inX;
    i_Y = inY;
    i_Z = inZ;
    weights = new MNC::Matrix(filter_size * filter_size, filter_count);
    file->read((char*)weights->data, sizeof(float) * weights->columns * weights->rows);
    bias = new MNC::Matrix(filter_count, 1);
    file->read((char*)bias->data, sizeof(float) * bias->columns * bias->rows);
    uint32_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    out = new MNC::Matrix(outX * outY * outZ, 1);
    outDer = new MNC::Matrix(outX * outY * outZ, 1);
}