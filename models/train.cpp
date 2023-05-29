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
    CommandLineParser parser(argc, argv,
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
    }

    String fd_modelPath = parser.get<String>("fd_model");
    String fr_modelPath = parser.get<String>("fr_model");

    String databasePath = parser.get<String>("database");

    float scoreThreshold = parser.get<float>("score_threshold");
    float nmsThreshold = parser.get<float>("nms_threshold");
    int topK = parser.get<int>("top_k");
    float scale = parser.get<float>("scale");

    /* [Initialize_FaceDetectorYN] */
    // Initialize FaceDetectorYN to the smart pointer detector
    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(fd_modelPath, "", Size(320, 320), scoreThreshold, nmsThreshold, topK);
    
    /* [initialize_FaceRecognizerSF] */
    // Initialize FaceRecognizerSF to the smart pointer faceRecognizer
    Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(fr_modelPath, "");

    // Create the ground truth folder if it does not exist
    String groundTruthFaces = "groundTruthFace";
    if (!filesystem::exists(groundTruthFaces)) {
        cout << "Creating database folder: " << groundTruthFaces << endl;
        filesystem::create_directory(groundTruthFaces);
    }

    // Read images and their names in database
    vector<Mat> images;
    vector<String> imageNames;
    for (auto& entry : filesystem::directory_iterator(databasePath)) {
        imageNames.push_back(entry.path().string());
        Mat image = imread(entry.path().string());
        if (!image.empty()) {
            images.push_back(image);
        }
    }

    /* Process all the images */
    for (const auto& image : images) {
        
        vector<Mat> out = detection(image, detector, faceRecognizer, scale);
        if (out.empty()) {
            cout << "No face detected in " << imageNames[&image - &images[0]] << endl;
            continue;
        }

        Mat feature = out[1];
        // Save the face to the ground truth folder with the name same as its file name
        String faceName = filesystem::path(imageNames[&image - &images[0]]).filename().string();
        imwrite(groundTruthFaces + "/" + faceName, feature);
    }
}