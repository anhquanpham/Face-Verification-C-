/*
    This file creates a database of ground truth labels of face in the database
    which are later used to verify the identity of the detected face.

    Input of this file is a database of images of face in the system. 
    These images are then processed by the face detection model to extract features.
    The used models here are the same models which are used in real time face verification application.

    The set of features of each face is then stored in a ground truth folder.
*/

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <vector>
#include <filesystem>
#include "detection.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Initialize parameters
    cv::CommandLineParser parser(argc, argv,
        "{help  h           |            | Print this message}"
        "{database d        | database   | Path to the database of ground truth face to verify}"
        "{scale sc          | 1.0        | Scale factor used to resize input video frames}"
        "{fd_model fd       | pretrained/yunet.onnx | Path to the model. Download yunet.onnx in https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet}"
        "{fr_model fr       | pretrained/sface.onnx | Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface}"
        "{score_threshold   | 0.9        | Filter out face of score < score_threshold}"
        "{nms_threshold     | 0.3        | Suppress bounding boxes of iou >= nms_threshold}"
        "{top_k             | 5000       | Keep top_k bounding boxes before NMS}"
    );
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    };

    cv::String fd_modelPath = parser.get<cv::String>("fd_model");
    cv::String fr_modelPath = parser.get<cv::String>("fr_model");

    cv::String databasePath = parser.get<cv::String>("database");

    float scoreThreshold = parser.get<float>("score_threshold");
    float nmsThreshold = parser.get<float>("nms_threshold");
    int topK = parser.get<int>("top_k");
    float scale = parser.get<float>("scale");

    // Initialize FaceDetectorYN to the smart pointer detector
    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(fd_modelPath, "", Size(320, 320), scoreThreshold, nmsThreshold, topK);
    
    // Initialize FaceRecognizerSF to the smart pointer faceRecognizer
    cv::Ptr<cv::FaceRecognizerSF> faceRecognizer = cv::FaceRecognizerSF::create(fr_modelPath, "");

    // Read images and their names in database
    std::vector<cv::Mat> images;
    std::vector<cv::String> imageNames;
    for (auto& entry : filesystem::directory_iterator(databasePath)) {
        imageNames.push_back(entry.path().string());
        cv::Mat image = imread(entry.path().string());
        if (!image.empty()) {
            images.push_back(image);
        }
    }

    // Setup the file to store the ground truth features and labels
    FileStorage fs("groundTruthFaces.yml", FileStorage::WRITE);

    /* Process all the images */
    vector<cv::Mat> features;
    vector<cv::String> labels;
    for (auto& image : images) {
        
        std::vector<cv::Mat> out = detection(image, detector, faceRecognizer, scale);
        cv::Mat feature = out[1];

        if (feature.empty()) {
            cout << "No face detected in " << imageNames[&image - &images[0]] << endl;
            continue;
        };

        // Save the feature to the ground truth feature vector
        features.push_back(feature);

        // Save the face name to the labels vector
        cv::String faceName = filesystem::path(imageNames[&image - &images[0]]).filename().stem().string();
        labels.push_back(faceName);
    }

    // Write the ground truth features and labels to the file
    fs << "features" << features;
    fs << "labels" << labels;
    fs.release();
    
    std::cout << "Feature extraction done." << std::endl;
    return 0;
}