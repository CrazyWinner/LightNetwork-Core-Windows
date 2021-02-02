#pragma once
#include <iostream>
#include <math.h>
#include <iomanip>
namespace MNC
{
  class Matrix
  {
  private:
    bool isTransposed = false;
    bool destroyAfter = true;
    bool isInversed = false;

  public:
    float *data;
    uint32_t rows, columns;
    Matrix(uint32_t r, uint32_t c);
    Matrix(uint32_t r, uint32_t c, float *arr);
    Matrix getSubMatrix(uint32_t r, uint32_t c, uint32_t id) const;
    ~Matrix();
    void operator+=(const Matrix &m);
    void operator+=(const float& m);
    Matrix operator*(const Matrix &m);
    float at(uint32_t i, uint32_t j) const;
    Matrix transpose() const;
    Matrix inverse() const;
    void doOperation(void (*op)(float &));
    Matrix operator-(const Matrix &m);
    void operator=(const Matrix &m);
    void hadamard(const Matrix &m);
    void operator*=(const float &f);
    void randomize();
    void fill(const float &a);
    void operator-=(const Matrix &m);
    void printDebug() const;
    int getIndex(uint32_t r, uint32_t c) const;
    void setTransposed(bool t);
    void setInversed(bool t);
    static Matrix fromArray(uint32_t r, uint32_t c, float *arr);
    void set(const uint32_t &r, const uint32_t &c, float a);
    Matrix convolve(const Matrix &m, int16_t paddingX, int16_t paddingY);
    float sum() const;
  };

} // namespace MNC