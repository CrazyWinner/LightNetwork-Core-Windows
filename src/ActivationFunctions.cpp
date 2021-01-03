#include "ActivationFunctions.h"

using namespace LightNetwork;
void RELU::activate(Matrix &m)
{
  m.doOperation([](float &in) {
    in = std::max((float)0, in);
  });
}

void RELU::derivative(Matrix &m)
{
  m.doOperation([](float &in) {
    in = in > 0 ? (float)1 : 0;
  });
}

RELU::~RELU()
{
  std::cout << "RELU DESTROYED" << std::endl;
}

void SIGMOID::activate(Matrix &m)
{
  m.doOperation([](float &in) {
    in = 1 / (1 + std::exp(-in));
  });
}
void SIGMOID::derivative(Matrix &m)
{
  m.doOperation([](float &in) {
    float sigmoid = 1 / (1 + std::exp(-in));
    in = sigmoid * (1 - sigmoid);
  });
}

SIGMOID::~SIGMOID()
{
  std::cout << "SIGMOID DESTROYED" << std::endl;
}

void LEAKY_RELU::activate(Matrix &m)
{
  m.doOperation([](float &in) {
    in = std::max(0.01f * in, in);
  });
}
void LEAKY_RELU::derivative(Matrix &m)
{
  m.doOperation([](float &in) {
    in = in > 0 ? 1 : 0.01f;
  });
}
LEAKY_RELU::~LEAKY_RELU()
{
  std::cout << "LEAKY_RELU DESTROYED" << std::endl;
}