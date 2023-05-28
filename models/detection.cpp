#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>

using namespace cv;
using namespace std;

static void visualize(Mat& input, int frame, Mat& faces, double fps, int thickness = 2) {
    /*
        This function draws bounding boxes and landmarks to the input image given the result of the face detection.
        Args:
            input (Mat): Input image
            frame (int): Frame number
            faces (Mat): Result of the face detection
            fps (double): Frames per second
            thickness (int): Thickness of the bounding box and landmark
        Output:
            None
    */

    std::string fpsString = cv::format("FPS : %.2f", (float)fps);
    if (frame >= 0)
        cout << "Frame " << frame << ", ";
    cout << "FPS: " << fpsString << endl;
    for (int i = 0; i < faces.rows; i++)
    {
        // Print results
        cout << "Face " << i
             << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
             << "box width: " << faces.at<float>(i, 2)  << ", box height: " << faces.at<float>(i, 3) << ", "
             << "score: " << cv::format("%.2f", faces.at<float>(i, 14))
             << endl;

        // Draw bounding box
        rectangle(input, Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), Scalar(0, 255, 0), thickness);
        // Draw landmarks
        circle(input, Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2, Scalar(255, 0, 0), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2, Scalar(0, 0, 255), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2, Scalar(0, 255, 0), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, Scalar(255, 0, 255), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, Scalar(0, 255, 255), thickness);
    }
    putText(input, fpsString, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
}

int main(int argc, char** argv)
{
    // Initialize parameters
    CommandLineParser parser(argc, argv,
        "{help  h           |            | Print this message}"
        "{image1 i1         |            | Path to the input image1. Omit for detecting through VideoCapture}"
        "{image2 i2         |            | Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm}"
        "{video v           | 0          | Path to the input video}"
        "{scale sc          | 1.0        | Scale factor used to resize input video frames}"
        "{fd_model fd       | yunet.onnx | Path to the model. Download yunet.onnx in https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet}"
        "{fr_model fr       | sface.onnx | Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface}"
        "{score_threshold   | 0.9        | Filter out faces of score < score_threshold}"
        "{nms_threshold     | 0.3        | Suppress bounding boxes of iou >= nms_threshold}"
        "{top_k             | 5000       | Keep top_k bounding boxes before NMS}"
        "{save s            | false      | Set true to save results. This flag is invalid when using camera}"
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

    bool save = parser.get<bool>("save");
    float scale = parser.get<float>("scale");

    // Similarity threshold
    double cosine_similar_thresh = 0.363;
    double l2norm_similar_thresh = 1.128;

    /* [Initialize_FaceDetectorYN] */
    // Initialize FaceDetectorYN to the smart pointer detector
    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(fd_modelPath, "", Size(320, 320), scoreThreshold, nmsThreshold, topK);
    //! [End initialize_FaceDetectorYN]

    TickMeter tm;

    // If input to detect is an image
    if (parser.has("image1"))
    {
        String input1 = parser.get<String>("image1");
        Mat image1 = imread(samples::findFile(input1));

        if (image1.empty()) {
            // Check if the input image is present
            std::cerr << "Cannot read image: " << input1 << std::endl;
            return 2;
        }

        // Resize image according to the scale factor to optimize the inference speed
        int imageWidth = int(image1.cols * scale);
        int imageHeight = int(image1.rows * scale);
        resize(image1, image1, Size(imageWidth, imageHeight));

        tm.start(); // Start timer

        /* [Inference] */
        // Set input size before inference
        detector->setInputSize(image1.size());

        Mat faces1;
        detector->detect(image1, faces1);
        if (faces1.rows < 1) {
            // Check if a face is detected
            std::cerr << "Cannot find a face in " << input1 << std::endl;
            return 1;
        }
        //! [End inference]

        tm.stop();
        // Draw results to the input image
        visualize(image1, -1, faces1, tm.getFPS());

        // Save results if save is true
        if (save)
        {
            cout << "Saving result.jpg...\n";
            imwrite("result.jpg", image1);
        }

        // Visualize results
        imshow("image1", image1);
        pollKey();  // handle UI events to show content

        if (parser.has("image2")) {
            /*
                If image2 is given, the program will try to find a face on image2 and run face recognition algorithm between image1 and image2
            */

            /* Face detection with image2 */
            String input2 = parser.get<String>("image2");
            Mat image2 = imread(samples::findFile(input2));
            if (image2.empty())
            {
                std::cerr << "Cannot read image2: " << input2 << std::endl;
                return 2;
            }

            tm.reset();
            tm.start();
            detector->setInputSize(image2.size());

            Mat faces2;
            detector->detect(image2, faces2);
            if (faces2.rows < 1)
            {
                std::cerr << "Cannot find a face in " << input2 << std::endl;
                return 1;
            }
            tm.stop();
            visualize(image2, -1, faces2, tm.getFPS());
            if (save)
            {
                cout << "Saving result2.jpg...\n";
                imwrite("result2.jpg", image2);
            }
            imshow("image2", image2);
            pollKey();

            /* [initialize_FaceRecognizerSF] */
            // Initialize FaceRecognizerSF
            Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(fr_modelPath, "");
            //! [End initialize_FaceRecognizerSF]


            /* [facerecognizer] */
            // Aligning and cropping facial image through the first face of faces detected.
            Mat aligned_face1, aligned_face2;
            faceRecognizer->alignCrop(image1, faces1.row(0), aligned_face1);
            faceRecognizer->alignCrop(image2, faces2.row(0), aligned_face2);

            // Run feature extraction with given aligned_face
            Mat feature1, feature2;
            faceRecognizer->feature(aligned_face1, feature1);
            feature1 = feature1.clone();
            faceRecognizer->feature(aligned_face2, feature2);
            feature2 = feature2.clone();
            //! [End facerecognizer]

            /* [Match the detected features between image1 and image2] */
            double cos_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_COSINE);
            double L2_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_NORM_L2);
            //! [End match]

            if (cos_score >= cosine_similar_thresh) {
                // Compare the two faces according to the cosine similarity threshold
                std::cout << "They have the same identity;";
            } else {
                std::cout << "They have different identities;";
            }

            std::cout << " Cosine Similarity: " << cos_score << ", threshold: " << cosine_similar_thresh << ". (higher value means higher similarity, max 1.0)\n";

            if (L2_score <= l2norm_similar_thresh) {
                // Compare the two faces according to the L2 norm similarity threshold
                std::cout << "They have the same identity;";
            } else {
                std::cout << "They have different identities.";
            }

            std::cout << " NormL2 Distance: " << L2_score << ", threshold: " << l2norm_similar_thresh << ". (lower value means higher similarity, min 0.0)\n";
        }
        cout << "Press any key to exit..." << endl;
        waitKey(0);
    }
    // If no image, then use video for detection
    else
    {
        int frameWidth, frameHeight;
        VideoCapture capture;
        std::string video = parser.get<string>("video");
        if (video.size() == 1 && isdigit(video[0]))
            capture.open(parser.get<int>("video"));
        else
            capture.open(samples::findFileOrKeep(video));  // keep GStreamer pipelines
        if (capture.isOpened())
        {
            frameWidth = int(capture.get(CAP_PROP_FRAME_WIDTH) * scale);
            frameHeight = int(capture.get(CAP_PROP_FRAME_HEIGHT) * scale);
            cout << "Video " << video
                << ": width=" << frameWidth
                << ", height=" << frameHeight
                << endl;
        }
        else
        {
            cout << "Could not initialize video capturing: " << video << "\n";
            return 1;
        }

        detector->setInputSize(Size(frameWidth, frameHeight));

        cout << "Press 'SPACE' to save frame, any other key to exit..." << endl;
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

            resize(frame, frame, Size(frameWidth, frameHeight));

            // Inference
            Mat faces;
            tm.start();
            detector->detect(frame, faces);
            tm.stop();

            Mat result = frame.clone();
            // Draw results on the input image
            visualize(result, nFrame, faces, tm.getFPS());

            // Visualize results
            imshow("Live", result);

            int key = waitKey(1);
            bool saveFrame = save;
            if (key == ' ') {
                saveFrame = true;
                key = 0;  // handled
            }

            if (saveFrame)
            {
                std::string frame_name = cv::format("frame_%05d.png", nFrame);
                std::string result_name = cv::format("result_%05d.jpg", nFrame);
                cout << "Saving '" << frame_name << "' and '" << result_name << "' ...\n";
                imwrite(frame_name, frame);
                imwrite(result_name, result);
            }

            ++nFrame;

            if (key > 0)
                break;
        }
        cout << "Processed " << nFrame << " frames" << endl;
    }
    cout << "Done." << endl;
    return 0;
}