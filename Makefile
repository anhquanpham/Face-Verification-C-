train: utils.cpp detection.cpp train.cpp
	g++ -o train utils.cpp detection.cpp train.cpp `pkg-config --cflags --libs opencv4`

main: utils.cpp main.cpp
	g++ -o main main.cpp utils.cpp `pkg-config --cflags --libs opencv4`