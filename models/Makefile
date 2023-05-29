train: utils.cpp detection.cpp train.cpp
	g++ -o train utils.cpp detection.cpp train.cpp `pkg-config --cflags --libs opencv4`

main: utils.cpp detection.cpp verification.cpp main.cpp
	g++ -o main utils.cpp detection.cpp verification.cpp main.cpp `pkg-config --cflags --libs opencv4`