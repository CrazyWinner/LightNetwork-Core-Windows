#include "Matrix3D.h"


Matrix3D::Matrix3D(uint32_t sX, uint32_t sY, uint32_t sZ)
{
    sizeX = sX;
    sizeY = sY;
    sizeZ = sZ;
    data = new float[sX * sY * sZ]();
}
Matrix3D::Matrix3D(uint32_t sX, uint32_t sY, uint32_t sZ, float *arr)
{
    data = arr;
    sizeX = sX;
    sizeY = sY;
    sizeZ = sZ;
    destroyAfter = false;
}

void Matrix3D::doOperation(void (*op)(float &))
{
    
    for (uint32_t i = 0; i < sizeX * sizeY * sizeZ; i++)
    {
        op(data[i]);
    }
}

Matrix3D::~Matrix3D()
{
    if (destroyAfter)
        delete[] data;
}
void Matrix3D::operator+=(const Matrix3D &m)
{
    if (sizeX != m.sizeX || sizeY != m.sizeY || sizeZ != m.sizeZ)
        throw std::runtime_error("+= operation error!");
    for (uint32_t i = 0; i < sizeX * sizeY * sizeZ; i++)
    {
        data[i] = data[i] + m.data[i];
    }
}

void Matrix3D::operator+=(const float &m)
{
    for (uint32_t i = 0; i < sizeX * sizeY * sizeZ; i++)
    {
        data[i] = data[i] + m;
    }
}

/*
WARNING: Transposed matrices will use the same pointer
DO NOT DELETE POINTER
*/
Matrix3D Matrix3D::transpose() const
{

    Matrix3D ret(sizeY, sizeX, sizeZ, data);
    ret.setTransposed(!isTransposed);
    ret.setInversed(isInversed);
    return ret;
}
/*
WARNING: Inversed matrices will use the same pointer
DO NOT DELETE POINTER
*/
Matrix3D Matrix3D::inverse() const
{
    Matrix3D ret(sizeX, sizeY, sizeZ, data);
    ret.setTransposed(isTransposed);
    ret.setInversed(!isInversed);
    return ret;
}

Matrix3D Matrix3D::convolve(const Matrix3D &m, int16_t paddingX, int16_t paddingY)
{
    Matrix3D r(sizeX - m.sizeX + 1 + (2 * paddingX), sizeY - m.sizeY + 1 + (2 * paddingY), 1);
    for (int64_t i = -paddingY; i < sizeY + paddingY - m.sizeY + 1; i++)
    {
        for (int64_t j = -paddingX; j < sizeX + paddingX - m.sizeX + 1; j++)
        {
            for (int64_t y = 0; y < m.sizeY; y++)
            {
                for (int64_t x = 0; x < m.sizeX; x++)
                {
                    if ((i + y < 0 || i + y >= sizeY || j + x < 0 || j + x >= sizeX))
                    {
                        continue;
                    }
                    r.data[r.getIndex(j + paddingX, i + paddingY, 0)] += m.data[m.getIndex(x, y, 0)] * data[getIndex(j + x, i + y, 0)];
                }
            }
        }
    }
    return r;
}

void Matrix3D::operator*=(const float &f)
{
    for (uint32_t i = 0; i < sizeX * sizeY * sizeZ; i++)
    {
        data[i] = data[i] * f;
    }
}

void Matrix3D::setTransposed(bool t)
{
    isTransposed = t;
}

void Matrix3D::setInversed(bool t)
{
    isInversed = t;
}

Matrix3D Matrix3D::operator-(const Matrix3D &m)
{
    if (sizeX != m.sizeX || sizeY != m.sizeY || sizeZ != m.sizeZ)
         throw std::runtime_error("- operation error!");
    Matrix3D r(sizeX, sizeY, sizeZ);
    for (uint32_t i = 0; i < sizeX * sizeY * sizeZ; i++)
    {
        r.data[i] = data[i] - m.data[i];
    }
    return r;
}

void Matrix3D::set(uint32_t x, uint32_t y, uint32_t z, float a)
{
    data[getIndex(x,y,z)] = a;
}

void Matrix3D::operator=(const Matrix3D &m)
{
    if (sizeX * sizeY * sizeZ != m.sizeX * m.sizeY * m.sizeZ)
    {
        delete data;
        data = new float[m.sizeX * m.sizeY * m.sizeZ];
    }
    sizeX = m.sizeX;
    sizeY = m.sizeY;
    sizeZ = m.sizeZ;
    isTransposed = m.isTransposed;
    isInversed = m.isInversed;
    memcpy(data, m.data, sizeof(float) * sizeX * sizeY * sizeZ);
}

Matrix3D Matrix3D::operator*(const Matrix3D &m)
{
    if (sizeX != m.sizeY || m.sizeZ != 1 || sizeZ != 1)
        throw std::runtime_error("* operation error!");
    

    Matrix3D r(m.sizeX, sizeY, 1);

    for (uint32_t y = 0; y < sizeY; y++)
    {
        for (uint32_t x = 0; x < m.sizeX; x++)
        {
            for (uint32_t k = 0; k < sizeX; k++)
            {
                r.data[r.getIndex(x, y, 0)] += data[getIndex(k, y, 0)] * m.data[m.getIndex(x, k, 0)];
            }
        }
    }
    return r;
}

void Matrix3D::setDestroyAfter(bool b){
    destroyAfter = b;
}

float Matrix3D::sum() const
{
    float ret = 0;
    for (uint32_t i = 0; i < sizeX * sizeY * sizeZ; i++)
    {
        ret += data[i];
    }
    return ret;
}

void Matrix3D::randomize()
{

    for (uint32_t i = 0; i < sizeX * sizeY * sizeZ; i++)
    {
        data[i] = ((double)rand() / (RAND_MAX + 1.0) * 0.2 - 0.1);
    }
}
/*

TEST THIS!!!!

*/
Matrix3D Matrix3D::get2DMatrixAt(uint32_t id) const
{
    Matrix3D ret(sizeX, sizeY, 1, data + ((size_t)id * sizeX * sizeY));
    ret.setTransposed(isTransposed);
    ret.setInversed(isInversed);
    return ret;
}

void Matrix3D::fill(const float &a)
{
    for (uint32_t i = 0; i < sizeX * sizeY * sizeZ; i++)
    {
        data[i] = a;
    }
}

void Matrix3D::hadamard(const Matrix3D &m)
{
    if (sizeX != m.sizeX || sizeY != m.sizeY || sizeZ != m.sizeZ)
        throw std::runtime_error("hadamard error!");
    for (uint32_t i = 0; i < sizeX * sizeY * sizeZ; i++)
    {
        data[i] = data[i] * m.data[i];
    }
}

void Matrix3D::printDebug(uint32_t z) const
{

    for (uint32_t y = 0; y < sizeY; y++)
    {
        for (uint32_t x = 0; x < sizeX; x++)
        {
            std::cout  << std::fixed << std::setprecision(3) << data[getIndex(x,y,z)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

Matrix3D Matrix3D::fromArray(uint32_t sX, uint32_t sY, uint32_t sZ, float *arr)
{
    Matrix3D m(sX, sY, sZ);
    memcpy(m.data, arr, sizeof(float) * sX * sY * sZ);
    return m;
}

void Matrix3D::operator-=(const Matrix3D &m)
{
    if (sizeX != m.sizeX || sizeY != m.sizeY || sizeZ != m.sizeZ)
        throw std::runtime_error("-= operation error!");
    for (uint32_t i = 0; i < sizeX * sizeY * sizeZ; i++)
    {
        data[i] = data[i] - m.data[i];
    }
}

float Matrix3D::at(uint32_t x, uint32_t y, uint32_t z) const
{
    return data[getIndex(x,y,z)];
}

int Matrix3D::getIndex(uint32_t x, uint32_t y, uint32_t z) const
{
    if (isInversed)
    {
        x = sizeX - 1 - x;
        y = sizeY - 1 - y;
    }
    if (!isTransposed)
        return z * sizeX * sizeY + y * sizeX + x;
    return z * sizeX * sizeY + x * sizeY + y;
}