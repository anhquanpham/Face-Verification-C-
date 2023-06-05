#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <fstream>
#include "utils.hpp"

using namespace cv;
using namespace std;

class Model {

private:
    cv::Ptr<FaceRecognizerSF> faceRecognizer; // face recognition model
    cv::Ptr<FaceDetectorYN> detector; // face detection model

public:

    Model(String fd_modelPath, String fr_modelPath, float scoreThreshold, float nmsThreshold, int topK) {
        /* This method initializes the models to be used */

        this->detector = FaceDetectorYN::create(fd_modelPath, "", Size(320, 320), scoreThreshold, nmsThreshold, topK);
        this->faceRecognizer = FaceRecognizerSF::create(fr_modelPath, "");
    }

    std::vector<cv::Mat> detection(cv::Mat image) {
        /*
            This method detects faces and their features from the input image.
            Args:
                image (Mat): Input image
            Output:
                faces (Mat): Result of the face detection
                feature (vector<Mat>): Feature of the detected face
        */        

        TickMeter tm;
        tm.reset();
        tm.start();
        this->detector->setInputSize(image.size());

        cv::Mat faces;
        this->detector->detect(image, faces);
        tm.stop();

        cv::Mat result = image.clone();
        visualize(result, -1, faces, tm.getFPS());

        // extract features
        vector<cv::Mat> out;
        out.push_back(result);

        if (!faces.empty()) {
            for (int i = 0; i < faces.rows; i++) {
                cv::Mat aligned_face_i, feature_i;
                this->faceRecognizer->alignCrop(result, faces.row(i), aligned_face_i);
                // Run feature extraction with given aligned_face
                this->faceRecognizer->feature(aligned_face_i, feature_i);
                out.push_back(feature_i);
            }
        }

        return out;
    }
    
    String verification(Mat feature, double cosine_similar_thresh, double l2norm_similar_thresh) {
        /*
            This method verifies the identity of the detected face according to the local databse.
            Args:
                feature (Mat): Feature of the detected face
                cosine_similar_thresh (double): Threshold of cosine similarity
                l2norm_similar_thresh (double): Threshold of L2 norm similarity
            Output:
                label (String): Name of the detected person
        */
        
        cout << "Detected face, verifying identity..." << endl;

        // Read features and labels from database
        String databasePath = "groundTruthFaces.yml";
        FileStorage fs(databasePath, FileStorage::READ);

        vector<Mat> features;
        vector<String> labels;
        // Read features as vector<Mat>
        cv::FileNode featuresNode = fs["features"];
        if (featuresNode.type() == cv::FileNode::SEQ) {
            featuresNode >> features;
        }
        // Read labels as vector<String>
        cv::FileNode labelsNode = fs["labels"];
        if (labelsNode.type() == cv::FileNode::SEQ) {
            labelsNode >> labels;
        }
        fs.release();
        
        // Loop over all the ground truth faces
        String label;
        double max_cos_score = 0;
        // double min_l2_score = 100;
        for (int i = 0; i < features.size(); i++) {
            // Compute cosine similarity
            double cos_score = this->faceRecognizer->match(feature, features[i], FaceRecognizerSF::DisType::FR_COSINE);
            // Compute L2 norm similarity
            double L2_score = this->faceRecognizer->match(feature, features[i], FaceRecognizerSF::DisType::FR_NORM_L2);
            // Compare cosine similarity and L2 norm similarity with the thresholds
            if (cos_score >= cosine_similar_thresh && L2_score <= l2norm_similar_thresh) {
                if (cos_score > max_cos_score) {
                    max_cos_score = cos_score;
                    // min_l2_score = L2_score;
                    label = labels[i];
                }
            }
        }
        return label;
    }
};

class Verification: public Model {
    /* This class verifies the identity of the detected face according to the local database. */

public:

    Verification(String fd_modelPath, String fr_modelPath, float scoreThreshold, float nmsThreshold, int topK): 
    Model(fd_modelPath, fr_modelPath, scoreThreshold, nmsThreshold, topK) {}

    cv::Mat get_image(cv::Mat &image, float scale) {
        /* This methods resize image according to the scale factor to optimize the inference speed */

        int imageWidth = int(image.cols * scale);
        int imageHeight = int(image.rows * scale);
        cv::resize(image, image, Size(imageWidth, imageHeight));
        return image;
    }

    void attendance_check(String label) {
        /* This method records the faces detected in a file */

        string filename = "attendance.txt";
        // check if the file exists
        ifstream file(filename);
        if (!file) {
            // create the file if it does not exist
            ofstream file(filename);
            file << label << endl;
            file.close();
        } else {
            // check if the label is already in the file
            string line;
            bool found = false;
            while (getline(file, line)) {
                if (line == label) {
                    found = true;
                    break;
                }
            }
            file.close();
            // append the label to the file if it is not in the file
            if (!found) {
                ofstream file(filename, ios_base::app);
                file << label << endl;
                file.close();
            }
        }
    }

    cv::Mat forward(cv::Mat &image, float scale, double cosine_similar_thresh, double l2norm_similar_thresh) {
        /* This method combines and runs a forward pass of detecting and verifying face */

        // Detect faces and extract features
        image = this->get_image(image, scale);
        std::vector<cv::Mat> out = this->detection(image);
        cv::Mat result = out[0];

        for (auto i = out.begin() + 1; i != out.end(); ++i) {
            cv::Mat feature = *i;
            String label = this->verification(feature, cosine_similar_thresh, l2norm_similar_thresh);
            if (!label.empty()) {
                attendance_check(label);
                show_label(result, label);
            } else {
                String err = "Your face is not recorded in our system";
                show_label(result, err);
            }
        }

        return result;
    }
};

int main(int argc, char** argv)
{
    // Initialize parameters
    CommandLineParser parser(argc, argv,
        "{help  h           |            | Print this message}"
        "{scale sc          | 1.0        | Scale factor used to resize input video frames}"
        "{fd_model fd       | pretrained/yunet.onnx | Path to the model. Download yunet.onnx in https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet}"
        "{fr_model fr       | pretrained/sface.onnx | Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface}"
        "{score_threshold   | 0.9        | Filter out faces of score < score_threshold}"
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

    float scoreThreshold = parser.get<float>("score_threshold");
    float nmsThreshold = parser.get<float>("nms_threshold");
    int topK = parser.get<int>("top_k");

    float scale = parser.get<float>("scale");

    // Similarity threshold
    double cosine_similar_thresh = 0.363;
    double l2norm_similar_thresh = 1.128;

    /* Capture camera */
    int frameWidth, frameHeight;
    VideoCapture capture(0);

    if (!capture.isOpened()) {
        std::cout << "Cannot open the video camera" << endl;
        cin.get(); //wait for any key press
        return -1;
    }

    frameWidth = int(capture.get(CAP_PROP_FRAME_WIDTH) * scale);
    frameHeight = int(capture.get(CAP_PROP_FRAME_HEIGHT) * scale);
    string window_name = "Face Verification";
    namedWindow(window_name); //create a window 

    std::cout << "Window size" << ": width=" << frameWidth << ", height=" << frameHeight << endl;

    std::cout << "Press any key to exit..." << endl;

    int nFrame = 0;
    while(true)
    {
        // Get frame
        Mat frame;
        if (!capture.read(frame))
        {
            cerr << "Can't grab frame! Stop\n";
            break;
        }

        // Detect and verify face
        Verification verification_instance(fd_modelPath, fr_modelPath, scoreThreshold, nmsThreshold, topK);
        Mat result = verification_instance.forward(frame, scale, cosine_similar_thresh, l2norm_similar_thresh);
        imshow(window_name, result);

        ++nFrame;

        int key = waitKey(1);
        if (key > 0)
            break;
    }
    std::cout << "Processed " << nFrame << " frames" << endl;

    std::cout << "Done." << endl;
    return 0;
}