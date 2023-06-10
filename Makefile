data: data_base.cpp
	g++ -std=c++17 -o data data_base.cpp `pkg-config --cflags --libs opencv4`

train: utils.cpp detection.cpp train.cpp
	g++ -std=c++17 -o train utils.cpp detection.cpp train.cpp `pkg-config --cflags --libs opencv4`

main: utils.cpp main.cpp
	g++ -o main main.cpp utils.cpp `pkg-config --cflags --libs opencv4`