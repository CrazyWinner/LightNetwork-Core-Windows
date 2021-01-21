#include "Layer.h"
class Conv2D : public Layer
{
private:
    uint16_t filter_size, filter_count, padding;
    uint16_t i_X, i_Y, i_Z;

public:
    Conv2D(uint16_t f_s, uint16_t f_c, uint16_t pd, Activation::ActivationType act, const float lr);
    MNC::Matrix feed_forward(MNC::Matrix &in);
    MNC::Matrix back_propagation(const MNC::Matrix &in, const MNC::Matrix &inDer, const MNC::Matrix &err);
    void init(uint16_t inX, uint16_t inY, uint16_t inZ);
    void getOutDimensions(uint16_t &outX, uint16_t &outY, uint16_t &outZ);
};
