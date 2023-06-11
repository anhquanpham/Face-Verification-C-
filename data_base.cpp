# include <opencv2/opencv.hpp>
# include <string>
# include <iostream>
# include<fstream>
# include <filesystem>
# include "utils.hpp"

using namespace cv;
using namespace std;


class Database {
private:
    cv::Ptr<FaceDetectorYN> detector;
    cv::Mat Captured_Image;

public:
    Database(std::string detector_path, float scoreThreshold, float nmsThreshold, int topK) {
        this->detector = FaceDetectorYN::create(detector_path, "", Size(320, 320), scoreThreshold, nmsThreshold, topK);
    }

    int capture_img() {
        /*
            Capture the image, save it to CAptured_Image attribute.
            Accept the image if face can be detected in it and require user to retake the photo other wise.
            Args:
                None
        */
        cv:: VideoCapture camera(0); // 0 indicate the default camera
        if (!camera.isOpened()) {
            std:: cout << "Failed to open the camera. Check available devices." << std::endl;
            // display the error message when the device is not found.
            return -1;
        }

        cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);
        cv::resizeWindow("Camera Feed", 800, 600);
        std::cout << "Press Enter to take photo " << std::endl;

        while (true) {
            cv::Mat frame;
            camera >> frame;
            if (frame.empty()) {
                std::cout << "Failed to capture frame." << std::endl;
                return -1;
            }

            // Flip the frame horizontally
            cv::flip(frame, frame, 1);

            // Display the camera feed
            cv::imshow("Camera Feed", frame);

            // Break the loop if user press any keys
            if (cv::waitKey(1)==13) {
                this->Captured_Image = frame;
                break;
            }

        }
    
        if (!this->is_face_detected(this->Captured_Image)) {
            std::cout << "Unable to detect face in the image. Please take another photo." << std::endl;
            this->Captured_Image.release();
            camera.release();
            this->capture_img();
        }
        else {
            std::cout << "Face detected." << std::endl;
            cv::waitKey(3000);
            camera.release();
            cv::destroyAllWindows();
        }

    return 0;
    
    }


    int save_img(cv::Mat frame, std::string label, std::string folder) {
        /*
            Save a image to the data base.
            Args:
                image (Mat): input image.
                lable (string): label of the image
                folder (string): name of folder that the image will be saved to
        */
        std::string file_path = "./" + folder + "/" + label + ".jpg";
        // chech if the folder path exist. If not, create one.
        if (!std::filesystem::exists("./" + folder)) {
            std::filesystem::create_directories("./" + folder);
        }

        bool result = cv::imwrite(file_path, frame);
        if (result) {
            std::cout << "Image saved." << std::endl;
        } 
        else {
            std::cout << "Failed to save the image." << std::endl;
            return -1;
        }
        return 0;
    }


    int is_face_detected(cv::Mat image) {
    /*
        Detect face from the image.
        Args:
            image (Mat): Input image
        Output:
            (bool) : true if face is detected and false, otherwise.
    */
        this->detector->setInputSize(image.size());
        cv::Mat faces;
        this->detector->detect(image, faces);

        return (!faces.empty());
    }


    int add_database() {
    /*
        Take a picture of user and save it to the data base.
        Args:
            None
    */
        this->capture_img();
        std::cout << "Do you want to save the image? (y/n): ";
        char choice;
        std::cin >> choice;
        if (choice == 'Y' || choice == 'y') {
            std::string folder = "database";
            std::string label;
            std::cout << "Enter the identity: ";
            std::cin >> label;
            this->save_img(this->Captured_Image, label, folder);
        }
        return 0;
    }
};


int main() {
    std::string detector_path = "pretrained/yunet.onnx";
    float score_threshold = 0.9;
    float nms_threshold = 0.3;
    int top_k = 5000;

    Database db(detector_path, score_threshold, nms_threshold, top_k);
    db.add_database();

    return 0;
}