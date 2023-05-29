#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include "utils.hpp"

using namespace std;
using namespace cv;

void visualize(Mat& input, int frame, Mat& faces, double fps) {
    /*
        This function draws bounding boxes, landmarks, and label to the input image given the result of the face detection.
        Args:
            input (Mat): Input image
            frame (int): Frame number
            faces (Mat): Result of the face detection
            fps (double): Frames per second
        Output:
            None
    */

    std::string fpsString = cv::format("FPS : %.2f", (float)fps);
    if (frame >= 0)
        cout << "Frame " << frame << ", ";
    cout << fpsString << endl;
    for (int i = 0; i < faces.rows; i++)
    {
        // Print results
        cout << "Face " << i
             << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
             << "box width: " << faces.at<float>(i, 2)  << ", box height: " << faces.at<float>(i, 3) << ", "
             << "score: " << cv::format("%.2f", faces.at<float>(i, 14))
             << endl;

        // Draw bounding box
        rectangle(input, Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), Scalar(0, 255, 0), 2);
        // Draw landmarks
        circle(input, Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2, Scalar(255, 0, 0), 2);
        circle(input, Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2, Scalar(0, 0, 255), 2);
        circle(input, Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2, Scalar(0, 255, 0), 2);
        circle(input, Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, Scalar(255, 0, 255), 2);
        circle(input, Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, Scalar(0, 255, 255), 2);
    }
    putText(input, fpsString, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
}

void show_label(Mat& input, String label) {
    /*
        This function draws label to the input image given the label.
        Args:
            input (Mat): Input image
            label (String): Label
            2 (int): 2 of the bounding box and landmark
        Output:
            None
    */
    // Log the result
    cout << "Detected face: " << label << endl;
    std::string text = "Welcome, " + label;
    // Calculate the position of the text
    int posX = input.cols / 2 - getTextSize(text, FONT_HERSHEY_SIMPLEX, 1, 2, 0).width / 2;
    int posY = input.rows * 2/3 + getTextSize(text, FONT_HERSHEY_SIMPLEX, 1, 2, 0).height / 2;

    putText(input, text, Point(0, 15), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
}