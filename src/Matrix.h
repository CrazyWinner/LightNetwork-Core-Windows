#pragma once
#include<iostream>
#include <math.h>
namespace LightNetwork
{
    class Matrix
    {
        private: 
          
          bool isTransposed = false;
          bool destroyAfter = true;
        public: 
          float* data;  
          int rows, columns;
          Matrix(int r, int c);
          Matrix(int r, int c, float* arr);
          ~Matrix();      
          void operator+=(const Matrix &m);
          Matrix operator*(const Matrix &m);
          float at(int i, int j);
          Matrix transpose();
          void doOperation(void (*op)(float&));
          Matrix operator-(const Matrix &m);
          void operator=(const Matrix &m);
          void hadamard(const Matrix &m);
          void operator*=(const float &f);
          void randomize();
          void fill();
          void operator-=(const Matrix &m);
          void printDebug() const;
          int getIndex (int r,int c) const;
          void setTransposed(bool t);
          static Matrix fromArray(int r, int c, float* arr);  
    };

} // namespace LightNetwork