#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>

using namespace std;
using namespace cv;

void visualize(Mat& input, int frame, Mat& faces, double fps);
void show_label(Mat& input, String label);

