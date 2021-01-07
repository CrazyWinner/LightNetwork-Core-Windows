#pragma once
#include "Matrix.h"

class Activation
{
public:
    virtual void activate(MNC::Matrix &m) = 0;
    virtual void derivative(MNC::Matrix &m) = 0;
};
