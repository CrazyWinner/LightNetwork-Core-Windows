#pragma once
#include "Activation.h"
#include "Matrix.h"
#include <algorithm>
#include "math.h"

class RELU : public Activation
{
public:
     virtual void activate(MNC::Matrix &m);
     virtual void derivative(MNC::Matrix &m);
     ~RELU();
};

class SIGMOID : public Activation
{
public:
     virtual void activate(MNC::Matrix &m);
     virtual void derivative(MNC::Matrix &m);
     ~SIGMOID();
};
class LEAKY_RELU : public Activation
{
public:
     virtual void activate(MNC::Matrix &m);
     virtual void derivative(MNC::Matrix &m);
     ~LEAKY_RELU();
};
