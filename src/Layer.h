#pragma once
#include "Matrix.h"
#include "Activation.h"
#include "ActivationFunctions.h"
#include <math.h>       /* exp */
#include<algorithm> 
namespace LightNetwork
{
    class Layer
    {
    private:
        float learning_rate=0.01;
        
    public:
        Activation* activator;
        Matrix* weights;
        Matrix* bias;
        Matrix* out;
        Matrix* outDer;
        int i_size,p_count;
        Layer(int i_s, int p_c, Activation* act);
        ~Layer();
        Matrix feed_forward(Matrix& in);
        void back_propagation(Matrix& in, Matrix& inDer, Matrix& err);    
    };

} // namespace LightNetwork