# Face Verification with opencv
Final Project ELEC2030 - VinUniversity

**Team:** Tran Quoc Bao, Tran Huy Hoang Anh, Pham Anh Quan

**Description:** Face verification using C++. The program is used to detect faces from the camera, then compare the detected faces with several faces in the database to get the identity of the person. The list of people in the database that appeared is stored in the file attendance.txt

## How to run

First, a database folder needs to be created. This folder stores the ground truth faces and the identities to compare. 

Example filepath: `database/identity.jpg` 

To take a picture of user, create a database if not exist and then add the picture to that database.
``` bash
make data
./data
```

To get the ground truth faces to compare from the database:
``` bash
make train
./train
```
Then, the program can be run by:
```bash
make main
./main
```

## References
[OpenCV Tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
