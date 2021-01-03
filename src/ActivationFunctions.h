#pragma once
#include "Activation.h"
#include "Matrix.h"
#include <algorithm>
#include "math.h"
namespace LightNetwork
{
     class RELU : public Activation
     {
     public:
        virtual void activate(Matrix& m);
        virtual void derivative(Matrix& m);
        ~RELU();
     };
    

     class SIGMOID : public Activation
     {
     public:
          virtual void activate(Matrix &m);
          virtual void derivative(Matrix &m);
          ~SIGMOID();

     };
     class LEAKY_RELU : public Activation
     {
     public:
          virtual void activate(Matrix &m);
          virtual void derivative(Matrix &m);
          ~LEAKY_RELU();

     };

} // namespace LightNetwork