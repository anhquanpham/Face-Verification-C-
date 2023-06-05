#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <vector>
#include "utils.hpp"

using namespace cv;
using namespace std;

std::vector<cv::Mat> detection(cv::Mat &image, cv::Ptr<FaceDetectorYN> detector, cv::Ptr<FaceRecognizerSF> faceRecognizer, float scale);