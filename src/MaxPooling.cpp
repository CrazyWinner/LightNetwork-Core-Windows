#include "MaxPooling.h"
MaxPooling::MaxPooling(uint32_t sizeX, uint32_t sizeY) : Layer(Activation::RELU, 0)
{
    pooling_size_X = sizeX;
    pooling_size_Y = sizeY;
}
void MaxPooling::init(uint32_t inX, uint32_t inY, uint32_t inZ)
{
    i_X = inX;
    i_Y = inY;
    i_Z = inZ;
    uint32_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    out = new MNC::Matrix(outX * outY * outZ, 1);
    outDer = new MNC::Matrix(outX * outY * outZ, 1);
    outDer->fill(1);
    inCoordX = new uint32_t[outX * outY * outZ]();
    inCoordY = new uint32_t[outX * outY * outZ]();
}


void MaxPooling::getOutDimensions(uint32_t &outX, uint32_t &outY, uint32_t &outZ)
{
    outX = i_X / pooling_size_X;
    if(outX * pooling_size_X < i_X) outX++;
    outY = i_Y / pooling_size_Y;
    if(outY * pooling_size_Y < i_Y) outY++;
    outZ = i_Z;
}

MaxPooling::~MaxPooling(){
    delete[] inCoordX;
    delete[] inCoordY;
}

MNC::Matrix MaxPooling::feed_forward(MNC::Matrix &in)
{
   uint32_t outX, outY, outZ;
   getOutDimensions(outX, outY, outZ);
   MNC::Matrix ret(outX * outY * outZ, 1);
   for(uint32_t z = 0; z < i_Z; z++){
       MNC::Matrix inSub = in.getSubMatrix(i_Y, i_X, z);
       MNC::Matrix retSub = ret.getSubMatrix(outY, outX, z);
       for(uint32_t i = 0; i < outY; i++){
           for(uint32_t j = 0; j < outX; j++){
              float m = inSub.at(i*pooling_size_Y, j*pooling_size_X);
              inCoordY[(z*outX*outY) + (i*outX) + j] = i * pooling_size_Y;
              inCoordX[(z*outX*outY) + (i*outX) + j] = j * pooling_size_X; 
              for(uint32_t y = 0; y < pooling_size_Y; y++){
                for(uint32_t x = 0; x < pooling_size_X; x++){
                   if((i * pooling_size_Y + y) >= inSub.rows || (j * pooling_size_X + x) >= inSub.columns) continue;
                   float a = inSub.at(i * pooling_size_Y + y, j * pooling_size_X + x);
                   if(a > m){
                     m = a;
                     inCoordY[(z*outX*outY) + (i*outX) + j] = i * pooling_size_Y + y;
                     inCoordX[(z*outX*outY) + (i*outX) + j] = j * pooling_size_X + x; 
                   }
                 }
              }
            retSub.set(i, j, m);   
           }
       }

   }
   *out = ret;
   return ret;
}

MNC::Matrix MaxPooling::back_propagation(const MNC::Matrix &in, const MNC::Matrix &inDer, const MNC::Matrix &err)
{
   uint32_t outX, outY, outZ;
   getOutDimensions(outX, outY, outZ);
   MNC::Matrix ret(i_X * i_Y * i_Z, 1);
   for(uint32_t z = 0; z < outZ; z++){
     MNC::Matrix retSub = ret.getSubMatrix(i_Y, i_X, z);
     MNC::Matrix errSub = err.getSubMatrix(outY, outX, z);
     for(uint32_t i = 0; i < outY; i++){
       for(uint32_t j = 0; j < outX; j++){
         retSub.set(inCoordY[(z*outX*outY) + (i*outX) + j], inCoordX[(z*outX*outY) + (i*outX) + j], errSub.at(i,j));
       }
     }
    }
    ret.hadamard(inDer);
    return ret;
}

void MaxPooling::save(std::ofstream* file){
    uint8_t type = MAX_POOLING;
    file->write((char*)&type, sizeof(type));
    file->write((char*)&pooling_size_X, sizeof(pooling_size_X));
    file->write((char*)&pooling_size_Y, sizeof(pooling_size_Y));
}

void MaxPooling::load(std::ifstream* file, uint32_t inX, uint32_t inY, uint32_t inZ)
{
    i_X = inX;
    i_Y = inY;
    i_Z = inZ;
    uint32_t outX, outY, outZ;
    getOutDimensions(outX, outY, outZ);
    out = new MNC::Matrix(outX * outY * outZ, 1);
    outDer = new MNC::Matrix(outX * outY * outZ, 1);
    outDer->fill(1);
    inCoordX = new uint32_t[outX * outY * outZ]();
    inCoordY = new uint32_t[outX * outY * outZ]();
}