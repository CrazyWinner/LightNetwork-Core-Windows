#include "Layer.h"
class Conv2D : public Layer
{
private:
    uint32_t filter_size, filter_count, padding;
    uint32_t i_X, i_Y, i_Z;

public:
    Conv2D(uint32_t f_s, uint32_t f_c, uint32_t pd, Activation::ActivationType act, const float lr);
    MNC::Matrix feed_forward(MNC::Matrix &in);
    MNC::Matrix back_propagation(const MNC::Matrix &in, const MNC::Matrix &inDer, const MNC::Matrix &err);
    void init(uint32_t inX, uint32_t inY, uint32_t inZ);
    void getOutDimensions(uint32_t &outX, uint32_t &outY, uint32_t &outZ);
};
