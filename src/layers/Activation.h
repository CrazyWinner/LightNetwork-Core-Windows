#pragma once
#include "../utils/Matrix3D.h"

class Activation
{
public:
    enum ActivationType :  uint8_t
    {
        RELU = 0,
        SIGMOID,
        LEAKY_RELU
    };
    static void activate(Matrix3D &m, ActivationType act)
    {
        switch (act)
        {
        case RELU:
        {
            m.doOperation([](float &in) {
                in = std::max((float)0, in);
            });
            break;
        }
        case SIGMOID:
        {
            m.doOperation([](float &in) {
                in = 1 / (1 + std::exp(-in));
            });
            break;
        }
        case LEAKY_RELU:
        {
            m.doOperation([](float &in) {
                in = std::max(0.01f * in, in);
            });
            break;
        }
        }
    };
    static void derivative(Matrix3D &m, int act)
    {
        switch (act)
        {
        case RELU:
        {
            m.doOperation([](float &in) {
                in = in > 0 ? (float)1 : 0;
            });
            break;
        }
        case SIGMOID:
        {
            m.doOperation([](float &in) {
                float sigmoid = 1 / (1 + std::exp(-in));
                in = sigmoid * (1 - sigmoid);
            });
            break;
        }
        case LEAKY_RELU:
        {
            m.doOperation([](float &in) {
                in = in > 0 ? 1 : 0.01f;
            });
            break;
        }
        }
    };
};
