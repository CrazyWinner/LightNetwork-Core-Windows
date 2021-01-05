#pragma once
#include <vector> 
#include "Layer.h"
namespace LightNetwork{
   class NeuralNetwork{  
       public:
         uint16_t i_count;
         std::vector<Layer*> layers;
         void reset();
         NeuralNetwork(uint16_t i_c);
         ~NeuralNetwork();
         void addLayer(uint16_t p_c, Activation* act, const float lr);
         Matrix guess(Matrix& in);
         void train(Matrix& in, Matrix& desired_result);
      


   };




}