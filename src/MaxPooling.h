#include "Layer.h"
class MaxPooling : public Layer
{
private:
    uint16_t pooling_size;
    uint16_t i_X, i_Y, i_Z;
    uint16_t* inCoordX;
    uint16_t* inCoordY;
public:
    MaxPooling(uint16_t size);
    MNC::Matrix feed_forward(MNC::Matrix &in);
    MNC::Matrix back_propagation(const MNC::Matrix &in, const MNC::Matrix &inDer, const MNC::Matrix &err);
    void init(uint16_t inX, uint16_t inY, uint16_t inZ);
    void getOutDimensions(uint16_t &outX, uint16_t &outY, uint16_t &outZ);
    ~MaxPooling();
};
