#include "Conv2D.h"
Conv2D::Conv2D(uint32_t fsx, uint32_t fsy, uint32_t f_c, uint32_t pd_x, uint32_t pd_y, Activation::ActivationType act, const float lr) : Layer(act, lr)
{
    filter_size_X = fsx;
    filter_size_Y = fsy;
    filter_count = f_c;
    padding_X = pd_x;
    padding_Y = pd_y;
}
void Conv2D::init(uint32_t inX, uint32_t inY, uint32_t inZ)
{
    i_X = inX;
    i_Y = inY;
    i_Z = inZ;
    weights = new Matrix3D(filter_size_X, filter_size_Y, filter_count);
    weights->randomize();
    bias = new Matrix3D(1, filter_count, 1);
    bias->fill(0.1);
    uint32_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    out = new Matrix3D(outX, outY, outZ);
    outDer = new Matrix3D(outX, outY, outZ);
}

void Conv2D::getOutDimensions(uint32_t &outX, uint32_t &outY, uint32_t &outZ)
{
    outX = i_X - filter_size_X + 1 + (2 * padding_X);
    outY = i_Y - filter_size_Y + 1 + (2 * padding_Y);
    outZ = i_Z * filter_count;
}

Matrix3D Conv2D::feed_forward(Matrix3D &in)
{
    uint32_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    Matrix3D r(outX, outY, outZ);
    for (uint32_t f = 0; f < filter_count; f++)
    {
        for (uint32_t z = 0; z < i_Z; z++)
        {
            Matrix3D inSub = in.get2DMatrixAt(z);
            Matrix3D o = inSub.convolve(weights->get2DMatrixAt(f), padding_X, padding_Y);
            o += bias->at(0, f, 0);
            r.get2DMatrixAt(f * i_Z + z) = o;
        }
    }
    *outDer = r;
    Activation::derivative(*outDer, activationType);
    Activation::activate(r, activationType);
    *out = r;

    return r;
}

Matrix3D Conv2D::back_propagation(Matrix3D &in, Matrix3D &err)
{
    Matrix3D ret(i_X, i_Y, i_Z);
    uint32_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    Matrix3D gradient(err.sizeX, err.sizeY, err.sizeZ);
    gradient = err;
    gradient *= learning_rate;
    for(uint32_t f = 0; f < filter_count; f++){
        for(uint32_t z = 0; z < i_Z; z++){
           Matrix3D inSub = in.get2DMatrixAt(z);
           Matrix3D gradientSub = gradient.get2DMatrixAt(f * i_Z + z);
           float db = gradientSub.sum();
           bias->set(0, f, 0, bias->at(0, f, 0) - db);
           Matrix3D dw = inSub.convolve(gradientSub, padding_X, padding_Y);
           weights->get2DMatrixAt(f) -= dw;
           int dx_padding_X = (i_X - outX + filter_size_X - 1) / 2;
           int dx_padding_Y = (i_Y - outY + filter_size_Y - 1) / 2;
           Matrix3D dx = gradientSub.convolve(weights->get2DMatrixAt(f), dx_padding_X, dx_padding_Y);
           ret.get2DMatrixAt(z) += dx;
        }
    }
    return ret;
}

void Conv2D::save(std::ofstream* file){
    uint8_t type = CONVOLUTIONAL;
    file->write((char*)&type, sizeof(type));
    file->write((char*)&activationType,sizeof(activationType));
    file->write((char*)&learning_rate, sizeof(learning_rate));
    file->write((char*)&filter_size_X, sizeof(filter_size_X));
    file->write((char*)&filter_size_Y, sizeof(filter_size_Y));
    file->write((char*)&filter_count, sizeof(filter_count));
    file->write((char*)&padding_X, sizeof(padding_X));
    file->write((char*)&padding_Y, sizeof(padding_Y));
    file->write((char*)weights->data, sizeof(float) * weights->sizeX * weights->sizeY * weights->sizeZ);
    file->write((char*)bias->data, sizeof(float) * bias->sizeX * bias->sizeY * bias->sizeZ);
}

void Conv2D::load(std::ifstream* file, uint32_t inX, uint32_t inY, uint32_t inZ)
{
    i_X = inX;
    i_Y = inY;
    i_Z = inZ;
    weights = new Matrix3D(filter_size_X, filter_size_Y, filter_count);
    file->read((char*)weights->data, sizeof(float) * weights->sizeX * weights->sizeY * weights->sizeZ);
    bias = new Matrix3D(1, filter_count, 1);
    file->read((char*)bias->data, sizeof(float) * bias->sizeX * bias->sizeY * bias->sizeZ);
    uint32_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    out = new Matrix3D(outX, outY, outZ);
    outDer = new Matrix3D(outX, outY, outZ);
}