
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(uint16_t i_X, uint16_t i_Y, uint16_t i_Z)
{
  inputX = i_X;
  inputY = i_Y;
  inputZ = i_Z;
}

NeuralNetwork::~NeuralNetwork()
{
  for (int i = 0; i < layers.size(); i++)
  {
    delete layers.at(i);
  }
  layers.clear();
}



void NeuralNetwork::addLayer(Layer* l)
{
  if (layers.empty())
  {
    l->init(inputX, inputY, inputZ);
    layers.push_back(l);
  }
  else
  {
    uint16_t outX, outY, outZ;
    layers.at(layers.size() - 1)->getOutDimensions(outX, outY, outZ);
    l->init(outX, outY, outZ);
    layers.push_back(l);
  }
}


MNC::Matrix NeuralNetwork::guess(MNC::Matrix &in)
{
  MNC::Matrix r = layers.at(0)->feed_forward(in);
  for (int i = 1; i < layers.size(); i++)
  {
    r = layers.at(i)->feed_forward(r);
  }
  return r;
}

void NeuralNetwork::train(MNC::Matrix &in, MNC::Matrix &desired_result)
{

  MNC::Matrix result = guess(in);
  MNC::Matrix err = result - desired_result;
  for (int i = layers.size() - 1; i > 0; i--)
  {
    layers.at(i)->back_propagation(*layers.at(i - 1)->out, *layers.at(i - 1)->outDer, err);
  }
  layers.at(0)->back_propagation(in, in, err);
}