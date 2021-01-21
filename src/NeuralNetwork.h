#pragma once
#include <vector>
#include "Activation.h"
#include "Layer.h"

class NeuralNetwork
{
public:
   uint16_t inputX, inputY, inputZ;
   std::vector<Layer *> layers;
   void reset();
   NeuralNetwork(uint16_t i_X, uint16_t i_Y, uint16_t i_Z);
   ~NeuralNetwork();
   void addLayer(Layer *l);
   MNC::Matrix guess(MNC::Matrix &in);
   void train(MNC::Matrix &in, MNC::Matrix &desired_result);
};
