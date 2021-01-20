#include "Layer.h"
class FullyConnected : public Layer
{
public:
    FullyConnected(uint16_t p_c, Activation::ActivationType act, const float lr);
    MNC::Matrix feed_forward(MNC::Matrix &in);
    void back_propagation(MNC::Matrix &in, MNC::Matrix &inDer, MNC::Matrix &err);
    void init(uint16_t inX, uint16_t inY, uint16_t inZ);
    void getOutDimensions(uint16_t& outX, uint16_t& outY, uint16_t& outZ);
};
