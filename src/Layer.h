#pragma once
#include "Matrix.h"
#include "Activation.h"
#include "ActivationFunctions.h"
#include <math.h> /* exp */
#include <algorithm>
namespace LightNetwork
{
    class Layer
    {
    public:
        float learning_rate;
        Activation *activator;
        Matrix *weights;
        Matrix *bias;
        uint16_t i_size, p_count;
        Matrix *out;
        Matrix *outDer;
        Layer(uint16_t i_s, uint16_t p_c, Activation *act, const float lr);
        ~Layer();
        Matrix feed_forward(Matrix &in);
        void back_propagation(Matrix &in, Matrix &inDer, Matrix &err);
    };

} // namespace LightNetwork