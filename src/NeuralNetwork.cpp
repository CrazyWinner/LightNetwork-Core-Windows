
#include "NeuralNetwork.h"
using namespace LightNetwork;

NeuralNetwork::NeuralNetwork(int i_c)
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

void NeuralNetwork::addLayer(int p_c, Activation *act, const float lr)
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


Matrix NeuralNetwork::guess(Matrix &in)
{
  Matrix r = layers.at(0)->feed_forward(in);
  for (int i = 1; i < layers.size(); i++)
  {
    r = layers.at(i)->feed_forward(r);
  }
  return r;
}

void NeuralNetwork::train(Matrix &in, Matrix &desired_result)
{

  Matrix result = guess(in);
  Matrix err = result - desired_result;
  for (int i = layers.size() - 1; i > 0; i--)
  {
    layers.at(i)->back_propagation(*layers.at(i - 1)->out, *layers.at(i - 1)->outDer, err);
  }
  layers.at(0)->back_propagation(in, in, err);
}