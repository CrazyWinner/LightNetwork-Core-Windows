#pragma once
#include "Matrix.h"

namespace LightNetwork
{
    class Activation
    {
    public:
        virtual void activate(Matrix& m) = 0;
        virtual void derivative(Matrix& m) = 0;
    };

} // namespace LightNetwork