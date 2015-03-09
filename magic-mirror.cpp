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

const string WINDOW_NAME = "Magic Mirror";

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

void trainFromCamera(RaspiCam_Cv& camera, CascadeClassifier& face_cascade, Ptr<FaceRecognizer>& model) {
	cout << "Training by capturing from camera" << endl;
	int trainingImageCount = 0;
	Mat captured;
	vector<Rect> faces;
	vector<Mat> trainingImages;
	vector<int> trainingLabels;
	while (trainingImageCount < 10) {
		try {
			camera.grab();
			camera.retrieve(captured);

			face_cascade.detectMultiScale(captured, faces, 1.1, 2, 0,Size(80, 80));
			if (faces.size() == 1) {
				trainingImageCount++;
				cout << "detected face for training" << endl;
				for (int i = 0; i < faces.size(); i++) {

					int x = faces[i].x;
					int y = faces[i].y;
					int h = y + faces[i].height;
					int w = x + faces[i].width;
					rectangle(captured, Point(x, y), Point(w, h),
							Scalar(0, 255, 0), 2, 8, 0);

					Mat cropedFace = captured(faces[i]);
					captured.copyTo(cropedFace);

					Size size(200, 200);
					Mat resizedCropedFace;
					resize(cropedFace, resizedCropedFace, size);

					trainingImages.push_back(resizedCropedFace);
					trainingLabels.push_back(0);
				}
			} else {
				cout << "no face detected" << endl;
			}
			imshow(WINDOW_NAME, captured);
			waitKey(30);
		} catch (cv::Exception& e) {
			cerr << "Error: " << e.msg << endl;
		}
	}
	cout << "Starting training..." << endl;
	model->train(trainingImages, trainingLabels);
	cout << "Training completed" << endl;
}

void trainFromCsv(const string& trainingDataCsvFile, Ptr<FaceRecognizer>& model) {
	vector<Mat> trainingImages;
	vector<int> trainingLabels;
	cout << "Training from CSV file '" << trainingDataCsvFile << "'" << endl;;
	try {
		read_csv(trainingDataCsvFile, trainingImages, trainingLabels);
	} catch (cv::Exception& e) {
		cerr << "Error opening file \"" << trainingDataCsvFile << ": " << e.msg << endl;
		exit(EXIT_FAILURE);
	}
	cout << "Starting training..." << endl;
	model->train(trainingImages, trainingLabels);
	cout << "Training completed" << endl;
}

string getNameForLabel(int label) {
	switch(label) {
		case 1:
			return "Joe";
		case 0:
			return "Lasse";
	}
	return "Unbekannt";
}

int main(int argc, const char *argv[]) {
	cout << "Opening Camera..." << endl;
	RaspiCam_Cv camera;
	camera.set(CV_CAP_PROP_FORMAT, CV_8UC1);

	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	//Ptr<FaceRecognizer> model = createEigenFaceRecognizer(0, 200.0);


	if (camera.open()) {
		CascadeClassifier face_cascade;
		if (face_cascade.load("/usr/local/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml")) {
			namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);

			if (argc < 2) {
				trainFromCamera(camera, face_cascade, model);
			} else {
				string trainingDataCsvFile = string(argv[1]);
				trainFromCsv(trainingDataCsvFile, model);
			}

			std::vector<Rect> faces;
			Mat captured;

			while (1) {
				try {
					camera.grab();
					camera.retrieve(captured);

					face_cascade.detectMultiScale(captured, faces, 1.1, 2, 0, Size(80, 80));
					if (faces.size() > 0) {
						cout << "detected " << faces.size() << " faces" << endl;
						for (int i = 0; i < faces.size(); i++) {

							int x = faces[i].x;
							int y = faces[i].y;
							int h = y + faces[i].height;
							int w = x + faces[i].width;
							rectangle(captured, Point(x, y), Point(w, h), Scalar(0, 255, 0), 2, 8, 0);


							Mat cropedFace = captured(faces[i]);
							//cropedFace.copyTo(captured);
							captured.copyTo(cropedFace);

							Size size(200,200);
							Mat resizedCropedFace;
							resize(cropedFace, resizedCropedFace, size);

							int predictedLabel = -1;
							double confidence = 0.0;

							//predictedLabel = model->predict(resizedCropedFace);
							model->predict(resizedCropedFace, predictedLabel, confidence);
							string result_message = format("Predicted label = %d, Confidence = %f.", predictedLabel, confidence);
							cout << result_message << endl;

							putText(captured, getNameForLabel(predictedLabel), Point(x, y - 5),
									FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 255, 0), 1, CV_AA);
						}
					} else {
						cout << "no face detected" << endl;
					}
					imshow(WINDOW_NAME, captured);
					waitKey(30);
				} catch (cv::Exception& e) {
					cerr << "Error: " << e.msg << endl;
				}
			}
			//cout << "Stopping camera..." << endl;
			//camera.release();
		} else {
			cout << "failed to load file /usr/local/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml" << endl;
			exit(EXIT_FAILURE);
		}
	} else {
		cout << "failed to open camera." << endl;
		exit(EXIT_FAILURE);
	}

	puts("Done.");
	return EXIT_SUCCESS;
}
