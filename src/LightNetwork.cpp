#include "LightNetwork.h"

using namespace LightNetwork;
void LightNetworkHelper::exportToFile(NeuralNetwork *network, std::string fileName)
{
  std::ofstream saveFile;
  saveFile.open(fileName + ".lnw");
  unsigned char layer_count = network->layers.size();
  saveFile << layer_count;
  for(int i = 0; i < layer_count; i++){
    Layer* l = network->layers.at(i);
    saveFile << (unsigned char) 2; // this should be activation
    saveFile << (unsigned int) l->weights->rows;
    saveFile << (unsigned int) l->weights->columns;
    for(int j = 0; j < l->weights->rows * l->weights->columns; j++){
    for(int k = 0 ; k < sizeof(float); k++){
     saveFile << *((unsigned char*) (&l->weights->data[j] + k));
     }
    }
    for(int j = 0; j < l->bias->rows * l->bias->columns; j++){
         for(int k = 0 ; k < sizeof(float); k++){
     saveFile << *((unsigned char*) (&l->bias->data[j] + k));
     }
    }
  }

  saveFile.close();
}
