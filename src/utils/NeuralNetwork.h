#pragma once
#include <vector>
#include "../layers/Activation.h"
#include "../layers/Layer.h"

class NeuralNetwork
{
public:
   uint32_t inputX, inputY, inputZ;
   std::vector<Layer *> layers;
   void reset();
   void init(uint32_t i_X, uint32_t i_Y, uint32_t i_Z);
   ~NeuralNetwork();
   void addLayer(Layer *l);
   Matrix3D guess(Matrix3D &in);
   void train(Matrix3D &in, Matrix3D &desired_result);
};
