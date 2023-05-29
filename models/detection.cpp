#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <vector>
#include "detection.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

std::vector<cv::Mat> detection(cv::Mat &image, cv::Ptr<FaceDetectorYN> detector, cv::Ptr<FaceRecognizerSF> faceRecognizer, float scale) {
    /*
        This function detects faces and their features from the input image.
        Args:
            image (Mat): Input image
            detector (Ptr<FaceDetectorYN>): FaceDetectorYN model instance
            faceRecognizer (Ptr<FaceRecognizerSF>): FaceRecognizerSF model instance
            scale (float): Scale factor used to resize input image
        Output:
            faces (Mat): Result of the face detection
            feature (Mat): Feature of the detected face
    */

    // Resize image according to the scale factor to optimize the inference speed
    int imageWidth = int(image.cols * scale);
    int imageHeight = int(image.rows * scale);
    cv::resize(image, image, Size(imageWidth, imageHeight));

    TickMeter tm;
    tm.reset();
    tm.start();
    detector->setInputSize(image.size());

    cv::Mat faces;
    detector->detect(image, faces);
    tm.stop();

    visualize(image, -1, faces, tm.getFPS());

    // extract features
    cv::Mat aligned_face, feature;
    if (!faces.empty()) {
        faceRecognizer->alignCrop(image, faces.row(0), aligned_face);
        // Run feature extraction with given aligned_face
        faceRecognizer->feature(aligned_face, feature);
        feature = feature.clone();
    } 

    std::vector<cv::Mat> out;
    out.push_back(faces);
    out.push_back(feature);
    return out;
}