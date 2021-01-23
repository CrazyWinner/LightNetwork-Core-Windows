#include "Minerva.h"
void Minerva::exportToFile(NeuralNetwork *network, std::string fileName)
{
  std::ofstream saveFile;
  saveFile.open(fileName + ".lnw", std::ios_base::out | std::ios_base::binary);
  uint16_t layer_count = network->layers.size();
  pushToFile(saveFile, &layer_count, sizeof(layer_count));
  pushToFile(saveFile, &network->inputX, sizeof(network->inputX));
  pushToFile(saveFile, &network->inputY, sizeof(network->inputY));
  pushToFile(saveFile, &network->inputZ, sizeof(network->inputZ));
  for (int i = 0; i < layer_count; i++)
  {
    Layer *l = network->layers.at(i);
    l->save(&saveFile);
    saveFile.flush();
  }

  saveFile.close();
}

NeuralNetwork *Minerva::importFromFile(std::string fileName)
{

  std::ifstream loadFile(fileName + ".lnw", std::ios_base::in | std::ios_base::binary);
  uint16_t layerCount;
  loadFile.read((char *)&layerCount, sizeof(layerCount));
  uint32_t inX, inY, inZ;
  loadFile.read((char *)&inX, sizeof(inX));
  loadFile.read((char *)&inY, sizeof(inY));
  loadFile.read((char *)&inZ, sizeof(inZ));
  NeuralNetwork *network = new NeuralNetwork(inX, inY, inZ);
  uint8_t type;
  
  while (layerCount != 0)
  {
    loadFile.read((char *)&type, sizeof(type));
    Layer *l;
    switch (type)
    {
    case Layer::FULLY_CONNECTED:
    {
      Activation::ActivationType activationType;
      float learning_rate;
      uint32_t p_count;
      loadFile.read((char *)&activationType, sizeof(activationType));
      loadFile.read((char *)&learning_rate, sizeof(learning_rate));
      loadFile.read((char *)&p_count, sizeof(p_count));
      l = new FullyConnected(p_count, activationType, learning_rate);
    }
    break;
    case Layer::CONVOLUTIONAL:
    {
      Activation::ActivationType activationType;
      float learning_rate;
      uint32_t filter_size, filter_count, padding;
      loadFile.read((char *)&activationType, sizeof(activationType));
      loadFile.read((char *)&learning_rate, sizeof(learning_rate));
      loadFile.read((char *)&filter_size, sizeof(filter_size));
      loadFile.read((char *)&filter_count, sizeof(filter_count));
      loadFile.read((char *)&padding, sizeof(padding));
      l = new Conv2D(filter_size, filter_count, padding, activationType, learning_rate);
    }
    break;
    case Layer::MAX_POOLING:
    {
      uint32_t poolingSize;
      loadFile.read((char *)&poolingSize, sizeof(poolingSize));
      l = new MaxPooling(poolingSize);
    }
    break;
    default:
      break;
    }
    l->load(&loadFile, inX, inY, inZ);
    l->getOutDimensions(inX, inY, inZ);
    network->layers.push_back(l);
    layerCount--;
  }
  return network;
}

void Minerva::pushToFile(std::ofstream &file, void *p, size_t size)
{
  file.write((char *)p, size);
}