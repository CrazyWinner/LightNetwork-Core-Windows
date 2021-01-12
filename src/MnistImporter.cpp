#include "MnistImporter.h"
#include <string>
MnistImporter::MnistImporter(char *imagesFile, char *labelsFile)
{

    imFile = new std::ifstream(imagesFile, std::ios_base::in | std::ios_base::binary);
    laFile = new std::ifstream(labelsFile, std::ios_base::in | std::ios_base::binary);
    uint32_t magicNumber, numOfImages;
    readMsbFirst(*imFile, (char *)&magicNumber, 4);
    readMsbFirst(*imFile, (char *)&numOfImages, 4);
    readMsbFirst(*imFile, (char *)&rows, 4);
    readMsbFirst(*imFile, (char *)&cols, 4);
}

MnistImporter::~MnistImporter()
{
    imFile->close();
    laFile->close();
    delete imFile;
    delete laFile;
}

MNC::Matrix MnistImporter::getInAt(uint32_t id)
{
    //std::cout << "Loading image" << id << std::endl;
    MNC::Matrix a((rows / resDivider) * (cols / resDivider),1);
    uint8_t read;
    for (uint32_t j = 0; j < rows / resDivider; j++)
    {
        for (uint32_t k = 0; k < cols / resDivider; k++)
        {
            float b = 0;
            for (uint32_t x = 0; x < resDivider; x++)
            {
                for (uint32_t y = 0; y < resDivider; y++)
                {
                    imFile->seekg(16 + 784 * id + ((j * resDivider + x) * cols) + (k * resDivider + y));
                    imFile->read((char *)&read, 1);
                    if(imFile->eof())std::cout << "eof" << std::endl;
                    b += read;
                }
            }
            b = b / 1024;
            a.set(j * (cols / resDivider) + k, 0, b);       
        }
    }

    return a;
}
MNC::Matrix MnistImporter::getOutAt(uint32_t id)
{
    //std::cout << "Loading label" << id << std::endl;
    laFile->seekg(8 + id);
    MNC::Matrix a(10, 1);
    unsigned char read;

    laFile->read((char *)&read, 1);
    for (uint8_t j = 0; j < 10; j++)
    {
        a.set(j, 0, j == read);
    }

    return a;
}

void MnistImporter::readMsbFirst(std::ifstream &file, char *ptr, size_t size)
{
    for (size_t i = size; i > 0; i--)
    {
        file.read(ptr + (i - 1), 1);
    }
}