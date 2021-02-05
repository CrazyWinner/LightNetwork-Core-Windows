#include "Flatten.h"

Flatten::Flatten() : Layer(Activation::RELU,0){

}

void Flatten::init(uint32_t inX, uint32_t inY, uint32_t inZ)
{
    iX = inX;
    iY = inY;
    iZ = inZ;
    out = new Matrix3D(1,  iX * iY * iZ, 1);
}

Matrix3D Flatten::feed_forward(Matrix3D &in)
{
    Matrix3D r = Matrix3D::fromArray(1, iX * iY * iZ, 1, in.data);
    //in.data = nullptr;
    *out = r;
    return r;
}

Matrix3D Flatten::back_propagation(const Matrix3D &in, const Matrix3D &err)
{
    Matrix3D ret = Matrix3D::fromArray(iX, iY, iZ, err.data);
    //err.data = nullptr;
    return ret;
}

void Flatten::getOutDimensions(uint32_t &outX, uint32_t &outY, uint32_t &outZ)
{
    outX = 1;
    outY = iX * iY * iZ;
    outZ = 1;
}

void Flatten::save(std::ofstream* file){
    uint8_t type = FLATTEN;
    file->write((char*)&type, sizeof(type));
}

void Flatten::load(std::ifstream* file, uint32_t inX, uint32_t inY, uint32_t inZ)
{
    iX = inX;
    iY = inY;
    iZ = inZ;
    out = new Matrix3D(1,  iX * iY * iZ, 1);

}
