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
    out = new Matrix3D(outX, outY, outZ);
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

Matrix3D MaxPooling::feed_forward(Matrix3D &in)
{
   uint32_t outX, outY, outZ;
   getOutDimensions(outX, outY, outZ);
   Matrix3D ret(outX, outY, outZ);
   for(uint32_t z = 0; z < i_Z; z++){
       Matrix3D inSub = in.get2DMatrixAt(z);
       Matrix3D retSub = ret.get2DMatrixAt(z);
       for(uint32_t i = 0; i < outY; i++){
           for(uint32_t j = 0; j < outX; j++){
              float m = inSub.at(j*pooling_size_X, i*pooling_size_Y, 0);
              inCoordY[(z*outX*outY) + (i*outX) + j] = i * pooling_size_Y;
              inCoordX[(z*outX*outY) + (i*outX) + j] = j * pooling_size_X; 
              for(uint32_t y = 0; y < pooling_size_Y; y++){
                for(uint32_t x = 0; x < pooling_size_X; x++){
                   if((i * pooling_size_Y + y) >= inSub.sizeY || (j * pooling_size_X + x) >= inSub.sizeX) continue;
                   float a = inSub.at(j * pooling_size_X + x, i * pooling_size_Y + y, 0);
                   if(a > m){
                     m = a;
                     inCoordY[(z*outX*outY) + (i*outX) + j] = i * pooling_size_Y + y;
                     inCoordX[(z*outX*outY) + (i*outX) + j] = j * pooling_size_X + x; 
                   }
                 }
              }
            retSub.set(j, i, 0, m);   
           }
       }

   }
   *out = ret;
   return ret;
}

Matrix3D MaxPooling::back_propagation(Matrix3D &in, Matrix3D &err)
{
   uint32_t outX, outY, outZ;
   getOutDimensions(outX, outY, outZ);
   Matrix3D ret(i_X, i_Y, i_Z);
   for(uint32_t z = 0; z < outZ; z++){
     Matrix3D retSub = ret.get2DMatrixAt(z);
     Matrix3D errSub = err.get2DMatrixAt(z);
     for(uint32_t i = 0; i < outY; i++){
       for(uint32_t j = 0; j < outX; j++){
         retSub.set(inCoordX[(z*outX*outY) + (i*outX) + j], inCoordY[(z*outX*outY) + (i*outX) + j], 0, errSub.at(j,i,0));
       }
     }
    }
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
    out = new Matrix3D(outX, outY, outZ);
    inCoordX = new uint32_t[outX * outY * outZ]();
    inCoordY = new uint32_t[outX * outY * outZ]();
}