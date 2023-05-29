#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include "utils.hpp"
#include "detection.hpp"
#include "verification.hpp"

using namespace cv;
using namespace std;

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

    TickMeter tm;

    /* Capture camera */
    int frameWidth, frameHeight;
    VideoCapture capture;
    std::string video = parser.get<string>("video");
    if (video.size() == 1 && isdigit(video[0]))
        capture.open(parser.get<int>("video"));
    else
        capture.open(samples::findFileOrKeep(video));  // keep GStreamer pipelines
    if (capture.isOpened()) {
        frameWidth = int(capture.get(CAP_PROP_FRAME_WIDTH) * scale);
        frameHeight = int(capture.get(CAP_PROP_FRAME_HEIGHT) * scale);
        cout << "Video " << video
            << ": width=" << frameWidth
            << ", height=" << frameHeight
            << endl;
    }
    else {
        cout << "Could not initialize video capturing: " << video << "\n";
        return 1;
    }

    cout << "Press 'SPACE' to save frame, any other key to exit..." << endl;

    /* [Initialize_FaceDetectorYN] */
    // Initialize FaceDetectorYN to the smart pointer detector
    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(fd_modelPath, "", Size(320, 320), scoreThreshold, nmsThreshold, topK);
    
    /* [initialize_FaceRecognizerSF] */
    // Initialize FaceRecognizerSF to the smart pointer faceRecognizer
    Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(fr_modelPath, "");

    int nFrame = 0;
    for (;;)
    {
        // Get frame
        Mat frame;
        if (!capture.read(frame))
        {
            cerr << "Can't grab frame! Stop\n";
            break;
        }

        // Detect and verify face
        vector<Mat> out = detection(frame, detector, faceRecognizer, scale);
        if (!out.empty()) {
            Mat result = out[0];
            Mat feature = out[1];
            String name = verification(feature, cosine_similar_thresh, l2norm_similar_thresh, faceRecognizer);
            if (!name.empty()) {
                show_label(result, name);
            } else {
                String label = "Your face is not recorded in our system";
                show_label(result, label);
            }
            imshow("Live", result);
        } else {
            imshow("Live", frame);
        }

        ++nFrame;

        int key = waitKey(1);
        if (key > 0)
            break;
    }
    cout << "Processed " << nFrame << " frames" << endl;

    cout << "Done." << endl;
    return 0;
}