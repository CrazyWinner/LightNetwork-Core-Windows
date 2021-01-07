#pragma once
#include <vector>
#include "Layer.h"

class NeuralNetwork
{
public:
   uint16_t i_count;
   std::vector<Layer *> layers;
   void reset();
   NeuralNetwork(uint16_t i_c);
   ~NeuralNetwork();
   void addLayer(uint16_t p_c, Activation *act, const float lr);
   MNC::Matrix guess(MNC::Matrix &in);
   void train(MNC::Matrix &in, MNC::Matrix &desired_result);
};
