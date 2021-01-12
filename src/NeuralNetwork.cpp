
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(uint16_t i_c)
{
  this->i_count = i_c;
}

NeuralNetwork::~NeuralNetwork()
{
  for (int i = 0; i < layers.size(); i++)
  {
    delete layers.at(i);
  }
  layers.clear();
}

void NeuralNetwork::addLayer(uint16_t p_c, Activation::ActivationType act, const float lr)
{
  if (layers.empty())
  {
    layers.push_back(new Layer(i_count, p_c, act, lr));
  }
  else
  {
    layers.push_back(new Layer(layers.at(layers.size() - 1)->p_count, p_c, act, lr));
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