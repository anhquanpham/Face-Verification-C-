# Face Verification with opencv
Final Project ELEC2030 - VinUniversity

**Team:** Tran Quoc Bao, Tran Huy Hoang Anh, Pham Anh Quan

**Description:** Face verification using C++. The program is used to detect faces from the camera, then compare the detected faces with several faces in the database to get the identity of the person.

## How to run

First, a database folder needs to be created. This folder stores the ground truth faces and the identities to compare. 

Example filepath: `/models/database/identity.jpg`

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