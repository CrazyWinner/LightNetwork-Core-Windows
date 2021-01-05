#include "Matrix.h"

using namespace LightNetwork;

Matrix::Matrix(int r, int c)
{
    this->rows = r;
    this->columns = c;
    data = new float[rows * columns]();
}
Matrix::Matrix(int r, int c, float *arr)
{
    data = arr;
    this->rows = r;
    this->columns = c;
    destroyAfter = false;
}

void Matrix::doOperation(void (*op)(float &))
{
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->columns; j++)
        {
            op(this->data[getIndex(i, j)]);
        }
    }
}

Matrix::~Matrix()
{
    if(destroyAfter)
    delete[] data;
}
void Matrix::operator+=(const Matrix &m)
{
    if (this->rows != m.rows || this->columns != m.columns)
        throw std::runtime_error("+= operation error!");
    for (int i = 0; i < this->rows * this->columns; i++)
    {
            this->data[i] = this->data[i] + m.data[i];
    }
}
/*
WARNING: Transposed matrices will use the same pointer
DO NOT DELETE POINTER
*/
Matrix Matrix::transpose()
{
    
    Matrix ret(this->columns, this->rows, this->data);
    ret.setTransposed(!this->isTransposed);
    return ret;
    
}

void Matrix::operator*=(const float &f)
{
    for (int i = 0; i < this->rows * this->columns; i++)
    {
            this->data[i] = this->data[i] * f;
    }
}

void Matrix::setTransposed(bool t){
    this->isTransposed = t;
}

Matrix Matrix::operator-(const Matrix &m)
{
    if (this->rows != m.rows || this->columns != m.columns)
        throw std::runtime_error("- operation error!");
    Matrix r(this->rows, this->columns);
    for (int i = 0; i < this->rows * this->columns; i++)
    {
            r.data[i] = this->data[i] - m.data[i];
    }
    return r;
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
    memcpy(this->data, m.data, sizeof(float) * m.rows * m.columns);
}

Matrix Matrix::operator*(const Matrix &m)
{
    if (this->columns != m.rows)
        throw std::runtime_error("* operation error!");
        
		Matrix r(this->rows, m.columns);
     
    
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < m.columns; j++)
        {
            for (int k = 0; k < this->columns; k++)
            {
                r.data[r.getIndex(i, j)] += this->data[getIndex(i, k)] * m.data[m.getIndex(k, j)];
            }
        }
    }
    return r;
}

void Matrix::randomize()
{
    
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->columns; j++)
        {
      //   this->data[getIndex(i, j)] = ((double)rand() / (RAND_MAX + 1.0)) - 0.5;
        this->data[getIndex(i, j)] = ((double)rand() / (RAND_MAX + 1.0) * 2 - 1);
        }
    }
}

void Matrix::fill()
{
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->columns; j++)
        {
            this->data[getIndex(i, j)] = 0.5;
        }
    }
}

void Matrix::hadamard(const Matrix &m)
{
    if (this->rows != m.rows || this->columns != m.columns)
        throw std::runtime_error("hamard operation error!");
    int b = this->rows * this->columns;
    for (int i = 0; i < b; i++)
    {
            this->data[i] = this->data[i] * m.data[i];
    }
}


void Matrix::printDebug() const
{

    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->columns; j++)
        {
            std::cout << data[getIndex(i, j)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

Matrix Matrix::fromArray(int r, int c, float* arr)
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

float Matrix::at(int i, int j)
{
    return data[getIndex(i, j)];
}

int Matrix::getIndex(int r, int c) const
{
    if(!isTransposed)return r * columns + c;
    return c * rows + r;
}