#pragma once
#include <vector>
#include "Activation.h"
#include "Layer.h"

class NeuralNetwork
{
public:
   uint32_t inputX, inputY, inputZ;
   std::vector<Layer *> layers;
   void reset();
   NeuralNetwork(uint32_t i_X, uint32_t i_Y, uint32_t i_Z);
   ~NeuralNetwork();
   void addLayer(Layer *l);
   MNC::Matrix guess(MNC::Matrix &in);
   void train(MNC::Matrix &in, MNC::Matrix &desired_result);
};
