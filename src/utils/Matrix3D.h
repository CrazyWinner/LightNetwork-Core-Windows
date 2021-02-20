#pragma once
#include <iostream>
#include <math.h>
#include <iomanip>

  class Matrix3D
  {
  private:
    bool isTransposed = false;
    bool destroyAfter = true;
    bool isInversed = false;

  public:
    void setDestroyAfter(bool b);
    float *data;
    uint32_t sizeX, sizeY, sizeZ;
    Matrix3D(uint32_t sX, uint32_t sY, uint32_t sZ);
    Matrix3D(uint32_t sX, uint32_t sY, uint32_t sZ, float *arr);
    Matrix3D get2DMatrixAt(uint32_t id) const;
    ~Matrix3D();
    void operator+=(const Matrix3D &m);
    void operator+=(const float& m);
    Matrix3D operator*(const Matrix3D &m);
    float at(uint32_t x, uint32_t y, uint32_t z) const;
    Matrix3D transpose() const;
    Matrix3D inverse() const;
    void doOperation(void (*op)(float &));
    Matrix3D operator-(const Matrix3D &m);
    void operator=(const Matrix3D &m);
    void hadamard(const Matrix3D &m);
    void operator*=(const float &f);
    void randomize();
    void fill(const float &a);
    void operator-=(const Matrix3D &m);
    void printDebug(uint32_t z) const;
    int getIndex(uint32_t x, uint32_t y, uint32_t z) const;
    void setTransposed(bool t);
    void setInversed(bool t);
    static Matrix3D fromArray(uint32_t sX, uint32_t sY, uint32_t sZ, float *arr);
    void set(uint32_t x, uint32_t y, uint32_t z, float a);
    Matrix3D convolve(const Matrix3D &m, int16_t paddingX, int16_t paddingY);
    float sum() const;
  };
