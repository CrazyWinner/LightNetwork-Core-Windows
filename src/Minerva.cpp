#include "Minerva.h"
#include "FullyConnected.h"
void Minerva::exportToFile(NeuralNetwork *network, std::string fileName)
{
  std::ofstream saveFile;
  saveFile.open(fileName + ".lnw", std::ios_base::out | std::ios_base::binary);
  unsigned char layer_count = network->layers.size();
  pushToFile(saveFile, &layer_count, 1);
  for(int i = 0; i < layer_count; i++){
    Layer* l = network->layers.at(i);
    unsigned char activation = 2;    //TODO: fix activation functions
    pushToFile(saveFile, &activation, 1);
    pushToFile(saveFile, &l->learning_rate, sizeof(float));
    pushToFile(saveFile, &l->weights->rows, 2);
    pushToFile(saveFile, &l->weights->columns, 2);
    pushToFile(saveFile, l->weights->data, sizeof(float) * l->weights->columns * l->weights->rows);
    pushToFile(saveFile, l->bias->data, sizeof(float) * l->bias->rows);
    saveFile.flush();
  }

  saveFile.close();
}

NeuralNetwork* Minerva::importFromFile(std::string fileName){
NeuralNetwork* network = new NeuralNetwork(1,1,1);
std::ifstream loadFile(fileName + ".lnw", std::ios_base::in | std::ios_base::binary);
unsigned char layerCount;
loadFile.read((char*)&layerCount, 1);
while(layerCount != 0){
   unsigned char activation;
   loadFile.read((char*)&activation, 1);
   float lr;
   loadFile.read((char*)&lr, sizeof(float));
   uint16_t rows, columns;
   loadFile.read((char*)&rows, 2);
   loadFile.read((char*)&columns, 2);
   //Layer* l = new FullyConnected(columns, rows, Activation::SIGMOID, lr);
   Layer* l = nullptr;
   loadFile.read((char*)l->weights->data, rows * columns * sizeof(float));
   loadFile.read((char*)l->bias->data, rows * sizeof(float));
   network->layers.push_back(l);
   l->weights->printDebug();
   l->bias->printDebug();
  layerCount--;

}
return network;
}

void Minerva::pushToFile(std::ofstream& file, void* p, size_t size){
  file.write((char*)p,size);
}