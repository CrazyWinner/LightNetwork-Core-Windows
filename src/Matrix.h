#pragma once
#include <iostream>
#include <math.h>
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
    uint16_t rows, columns;
    Matrix(uint16_t r, uint16_t c);
    Matrix(uint16_t r, uint16_t c, float *arr);
    Matrix getSubMatrix(uint16_t r, uint16_t c, uint16_t id) const;
    ~Matrix();
    void operator+=(const Matrix &m);
    void operator+=(const float& m);
    Matrix operator*(const Matrix &m);
    float at(uint16_t i, uint16_t j) const;
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
    int getIndex(uint16_t r, uint16_t c) const;
    void setTransposed(bool t);
    void setInversed(bool t);
    static Matrix fromArray(uint16_t r, uint16_t c, float *arr);
    void set(const uint16_t &r, const uint16_t &c, float a);
    Matrix convolve(const Matrix &m, uint16_t padding);
  };

} // namespace MNC