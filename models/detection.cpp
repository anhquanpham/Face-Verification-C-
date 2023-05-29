#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <vector>
#include "utils.hpp"
#include "detection.hpp"

using namespace cv;
using namespace std;

vector<Mat> detection(Mat &image, Ptr<FaceDetectorYN> detector, Ptr<FaceRecognizerSF> faceRecognizer, float scale) {

    // Resize image according to the scale factor to optimize the inference speed
    int imageWidth = int(image.cols * scale);
    int imageHeight = int(image.rows * scale);
    resize(image, image, Size(imageWidth, imageHeight));

    TickMeter tm;
    tm.reset();
    tm.start();
    detector->setInputSize(image.size());

    Mat faces;
    detector->detect(image, faces);
    tm.stop();

    visualize(image, -1, faces, tm.getFPS());

    // extract features
    Mat aligned_face, feature;
    faceRecognizer->alignCrop(image, faces.row(0), aligned_face);
    // Run feature extraction with given aligned_face
    faceRecognizer->feature(aligned_face, feature);
    feature = feature.clone();

    vector<Mat> out = {image, feature};
    return out;
}