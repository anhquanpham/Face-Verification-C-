#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;

String verification(Mat feature, double cosine_similar_thresh, double l2norm_similar_thresh, Ptr<FaceRecognizerSF> faceRecognizer);

