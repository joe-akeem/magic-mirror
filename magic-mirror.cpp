//============================================================================
// Name        : magic-mirror2.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>

using namespace std;

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <raspicam/raspicam_cv.h>

using namespace cv;
using namespace raspicam;

void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels,
		char separator = ';') {
	puts("Reading CSV file...");
	ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message =
				"No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
	puts("Done reading CSV file.");
}

void train(const string& trainingDataCsvFile, Ptr<FaceRecognizer>& model) {
	vector<Mat> trainingImages;
	vector<int> trainingLabels;
	puts("starting training...");
	try {
		read_csv(trainingDataCsvFile, trainingImages, trainingLabels);
	} catch (cv::Exception& e) {
		cerr << "Error opening file \"" << trainingDataCsvFile << ": " << e.msg
				<< endl;
		exit(1);
	}

	model->train(trainingImages, trainingLabels);
	puts("training completed.");
}

int main(int argc, const char *argv[]) {
	if (argc < 2) {
		cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
		exit(1);
	}

	string output_folder = ".";
	if (argc == 3) {
		output_folder = string(argv[2]);
	}

	string trainingDataCsvFile = string(argv[1]);
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	train(trainingDataCsvFile, model);

	cout << "Opening Camera..." << endl;
	RaspiCam_Cv camera;
	camera.set(CV_CAP_PROP_FORMAT, CV_8UC1);
	if (camera.open()) {
		CascadeClassifier face_cascade;
		if (face_cascade.load("/usr/local/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml")) {
			std::vector<Rect> faces;
			Mat captured;
			while (1) {
				camera.grab();
				camera.retrieve(captured);
				//imwrite("camptured.jpg", image);

				face_cascade.detectMultiScale(captured, faces, 1.1, 2, 0, Size(80, 80));
				if (faces.size() > 0) {
					cout << "detected " << faces.size() << " faces" << endl;
					for (int i = 0; i < faces.size(); i++) {

						Mat cropedFace = captured(faces[i]);
						cropedFace.copyTo(captured);

						Size size(200,200);
						Mat resizedCropedFace;
						resize(cropedFace, resizedCropedFace, size);

						int predictedLabel = model->predict(resizedCropedFace);
						string result_message = format("Predicted label = %d.", predictedLabel);
						cout << result_message << endl;
					}
				} else {
					cout << "no face detected" << endl;
				}
			}
			//cout << "Stopping camera..." << endl;
			//camera.release();
		} else {
			cout << "failed to load file /usr/local/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml" << endl;
		}
	} else {
		cout << "failed to open camera." << endl;
	}

	puts("Done.");
	return EXIT_SUCCESS;
}
