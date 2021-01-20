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
    ~Matrix();
    void operator+=(const Matrix &m);
    Matrix operator*(const Matrix &m);
    float at(uint16_t i, uint16_t j);
    Matrix transpose();
    Matrix inverse();
    void doOperation(void (*op)(float &));
    Matrix operator-(const Matrix &m);
    void operator=(const Matrix &m);
    void hadamard(const Matrix &m);
    void operator*=(const float &f);
    void randomize();
    void fill();
    void operator-=(const Matrix &m);
    void printDebug() const;
    int getIndex(uint16_t r, uint16_t c) const;
    void setTransposed(bool t);
    void setInversed(bool t);
    static Matrix fromArray(uint16_t r, uint16_t c, float *arr);
    void set(const uint16_t& r, const uint16_t& c, float a);
  };

} // namespace MNC