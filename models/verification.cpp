#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <filesystem>
#include "verification.hpp"

using namespace cv;
using namespace std;

String verification(Mat feature, double cosine_similar_thresh, double l2norm_similar_thresh, Ptr<FaceRecognizerSF> faceRecognizer) {
    /*
        This function verifies the identity of the detected face according to the local databse.
        Args:
            feature (Mat): Feature of the detected face
            cosine_similar_thresh (double): Threshold of cosine similarity
            l2norm_similar_thresh (double): Threshold of L2 norm similarity
            faceRecognizer (Ptr<FaceRecognizerSF>): FaceRecognizerSF model instance
        Output:
            label (String): Name of the detected person
    */
    
    cout << "Detected face, verifying identity..." << endl;
    
    // Path to the database of ground truth features
    String databasePath = "groundTruthFace";
    
    // Loop over all the ground truth faces
    String label;
    double max_cos_score = 0;
    // double min_l2_score = 100;
    for (const auto& entry : filesystem::directory_iterator(databasePath)) {
        // Read the ground truth feature
        Mat groundTruthFeature = imread(entry.path(), IMREAD_UNCHANGED);
        // Compute cosine similarity
        double cos_score = faceRecognizer->match(feature, groundTruthFeature, FaceRecognizerSF::DisType::FR_COSINE);
        // Compute L2 norm similarity
        double L2_score = faceRecognizer->match(feature, groundTruthFeature, FaceRecognizerSF::DisType::FR_NORM_L2);
        // Compare cosine similarity and L2 norm similarity with the thresholds
        if (cos_score >= cosine_similar_thresh && L2_score <= l2norm_similar_thresh) {
            if (cos_score > max_cos_score) {
                max_cos_score = cos_score;
                // min_l2_score = L2_score;
                label = entry.path().stem().string();
            }
        }
    }
    return label;
}