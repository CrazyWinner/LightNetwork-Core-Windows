#include "Matrix.h"

using namespace MNC;

Matrix::Matrix(uint16_t r, uint16_t c)
{
    this->rows = r;
    this->columns = c;
    data = new float[rows * columns]();
}
Matrix::Matrix(uint16_t r, uint16_t c, float *arr)
{
    data = arr;
    this->rows = r;
    this->columns = c;
    destroyAfter = false;
}

void Matrix::doOperation(void (*op)(float &))
{
    for (uint32_t i = 0; i < this->rows * this->columns; i++)
    {
        op(this->data[i]);
    }
}

Matrix::~Matrix()
{
    if (destroyAfter)
        delete[] data;
}
void Matrix::operator+=(const Matrix &m)
{
    if (this->rows != m.rows || this->columns != m.columns)
        throw std::runtime_error("+= operation error!");
    for (uint32_t i = 0; i < this->rows * this->columns; i++)
    {
        this->data[i] = this->data[i] + m.data[i];
    }
}

void Matrix::operator+=(const float &m)
{
    for (uint32_t i = 0; i < this->rows * this->columns; i++)
    {
        this->data[i] = this->data[i] + m;
    }
}

/*
WARNING: Transposed matrices will use the same pointer
DO NOT DELETE POINTER
*/
Matrix Matrix::transpose() const
{

    Matrix ret(this->columns, this->rows, this->data);
    ret.setTransposed(!this->isTransposed);
    ret.setInversed(this->isInversed);
    return ret;
}
/*
WARNING: Inversed matrices will use the same pointer
DO NOT DELETE POINTER
*/
Matrix Matrix::inverse() const
{
    Matrix ret(this->rows, this->columns, this->data);
    ret.setTransposed(this->isTransposed);
    ret.setInversed(!this->isInversed);
    return ret;
}

Matrix Matrix::convolve(const Matrix &m, uint16_t padding)
{
    Matrix r(rows - m.rows + 1 + (2 * padding), columns - m.columns + 1 + (2 * padding));
    for (int32_t i = -padding; i < rows + padding - m.rows + 1; i++)
    {
        for (int32_t j = -padding; j < columns + padding - m.columns + 1; j++)
        {
            for (int32_t y = 0; y < m.rows; y++)
            {
                for (int32_t x = 0; x < m.columns; x++)
                {
                    if ((i + y < 0 || i + y >= rows || j + x < 0 || j + x >= columns))
                    {
                        continue;
                    }
                    r.data[r.getIndex(i + padding, j + padding)] += m.data[m.getIndex(y, x)] * data[getIndex(i + y, j + x)];
                }
            }
        }
    }
    return r;
}

void Matrix::operator*=(const float &f)
{
    for (uint32_t i = 0; i < this->rows * this->columns; i++)
    {
        this->data[i] = this->data[i] * f;
    }
}

void Matrix::setTransposed(bool t)
{
    this->isTransposed = t;
}

void Matrix::setInversed(bool t)
{
    this->isInversed = t;
}

Matrix Matrix::operator-(const Matrix &m)
{
    if (this->rows != m.rows || this->columns != m.columns)
        throw std::runtime_error("- operation error!");
    Matrix r(this->rows, this->columns);
    for (uint32_t i = 0; i < this->rows * this->columns; i++)
    {
        r.data[i] = this->data[i] - m.data[i];
    }
    return r;
}

void Matrix::set(const uint16_t &r, const uint16_t &c, float a)
{
    data[getIndex(r, c)] = a;
}

void Matrix::operator=(const Matrix &m)
{
    if (this->rows * this->columns != m.rows * m.columns)
    {
        delete this->data;
        this->data = new float[m.rows * m.columns];
    }
    this->rows = m.rows;
    this->columns = m.columns;
    this->isTransposed = m.isTransposed;
    this->isInversed = m.isInversed;
    memcpy(this->data, m.data, sizeof(float) * m.rows * m.columns);
}

Matrix Matrix::operator*(const Matrix &m)
{
    if (this->columns != m.rows)
    {
        std::cout << "ERR: " << this->rows << ":" << this->columns << " * " << m.rows << ":" << m.columns << std::endl;
        throw std::runtime_error("* operation error!");
    }

    Matrix r(this->rows, m.columns);

    for (uint16_t i = 0; i < this->rows; i++)
    {
        for (uint16_t j = 0; j < m.columns; j++)
        {
            for (uint16_t k = 0; k < this->columns; k++)
            {
                r.data[r.getIndex(i, j)] += this->data[getIndex(i, k)] * m.data[m.getIndex(k, j)];
            }
        }
    }
    return r;
}

float Matrix::sum() const
{
    float ret = 0;
    for (uint32_t i = 0; i < this->rows * this->columns; i++)
    {
        ret += this->data[i];
    }
    return ret;
}

void Matrix::randomize()
{

    for (uint32_t i = 0; i < this->rows * this->columns; i++)
    {
        this->data[i] = ((double)rand() / (RAND_MAX + 1.0) * 0.2 - 0.1);
    }
}
/*

TEST THIS!!!!

*/
Matrix Matrix::getSubMatrix(uint16_t r, uint16_t c, uint16_t id) const
{
    Matrix ret(r, c, this->data + ((size_t)id * r * c));
    ret.setTransposed(this->isTransposed);
    ret.setInversed(this->isInversed);
    return ret;
}

void Matrix::fill(const float &a)
{
    for (uint32_t i = 0; i < this->rows * this->columns; i++)
    {
        this->data[i] = a;
    }
}

void Matrix::hadamard(const Matrix &m)
{
    if (this->rows != m.rows || this->columns != m.columns)
        throw std::runtime_error("hamard operation error!");
    for (uint32_t i = 0; i < this->rows * this->columns; i++)
    {
        this->data[i] = this->data[i] * m.data[i];
    }
}

void Matrix::printDebug() const
{

    for (uint16_t i = 0; i < this->rows; i++)
    {
        for (uint16_t j = 0; j < this->columns; j++)
        {
            std::cout  << std::fixed << std::setprecision(3) << data[getIndex(i, j)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

Matrix Matrix::fromArray(uint16_t r, uint16_t c, float *arr)
{
    Matrix m(r, c);
    memcpy(m.data, arr, sizeof(float) * r * c);
    return m;
}

void Matrix::operator-=(const Matrix &m)
{
    if (this->rows != m.rows || this->columns != m.columns)
        throw std::runtime_error("-= operation error!");
    for (int i = 0; i < this->rows * this->columns; i++)
    {
        this->data[i] = this->data[i] - m.data[i];
    }
}

float Matrix::at(uint16_t i, uint16_t j) const
{
    return data[getIndex(i, j)];
}

int Matrix::getIndex(uint16_t r, uint16_t c) const
{
    if (isInversed)
    {
        r = rows - 1 - r;
        c = columns - 1 - c;
    }
    if (!isTransposed)
        return r * columns + c;
    return c * rows + r;
}