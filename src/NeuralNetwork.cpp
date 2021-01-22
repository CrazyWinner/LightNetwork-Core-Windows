
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(uint32_t i_X, uint32_t i_Y, uint32_t i_Z)
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

void NeuralNetwork::addLayer(Layer *l)
{
  if (layers.empty())
  {
    l->init(inputX, inputY, inputZ);
    layers.push_back(l);
  }
  else
  {
    uint32_t outX, outY, outZ;
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
  for (size_t i = layers.size() - 1; i > 0; i--)
  {
    err = layers.at(i)->back_propagation(*layers.at(i - 1)->out, *layers.at(i - 1)->outDer, err);
  }
  layers.at(0)->back_propagation(in, in, err);
}