cmake_minimum_required(VERSION 2.8.9)
project (Minerva)

file(GLOB SOURCES "src/labels/*.cpp")
file(GLOB SOURCES "src/utils/*.cpp")
file(GLOB SOURCES "src/mnist/*.cpp")

add_executable(main src/main.cpp
src/layers/Conv2D.cpp
src/layers/Flatten.cpp
src/layers/FullyConnected.cpp
src/layers/MaxPooling.cpp
src/mnist/MnistImporter.cpp
src/utils/HighResClock.cpp
src/utils/Matrix3D.cpp
src/utils/Minerva.cpp
src/utils/NeuralNetwork.cpp
)

add_custom_target(run
    COMMAND main.exe
    DEPENDS main.exe
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)