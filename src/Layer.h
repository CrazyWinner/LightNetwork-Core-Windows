#pragma once
#include "Matrix.h"
#include "Activation.h"
#include "ActivationFunctions.h"
#include <math.h> /* exp */
#include <algorithm>

class Layer
{
public:
    float learning_rate;
    Activation *activator;
    MNC::Matrix *weights;
    MNC::Matrix *bias;
    uint16_t i_size, p_count;
    MNC::Matrix *out;
    MNC::Matrix *outDer;
    Layer(uint16_t i_s, uint16_t p_c, Activation *act, const float lr);
    ~Layer();
    MNC::Matrix feed_forward(MNC::Matrix &in);
    void back_propagation(MNC::Matrix &in, MNC::Matrix &inDer, MNC::Matrix &err);
};
