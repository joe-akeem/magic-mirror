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
const string CROPED_WINDOW_NAME = "Croped Face";

vector<string> subjectNames;

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

Mat captureSingleImage(RaspiCam_Cv& camera, CascadeClassifier& face_cascade) {
	cout << "Capturing single image..." << endl;
	Mat captured;
	vector<Rect> faces;
	while (true) {
		try {
			camera.grab();
			camera.retrieve(captured);
			imshow(WINDOW_NAME, captured);
			waitKey(30);
			face_cascade.detectMultiScale(captured, faces, 1.1, 2, 0,Size(80, 80));
			if (faces.size() == 1) {
				cout << "Detected face!" << endl;
				int x = faces[0].x;
				int y = faces[0].y;
				int h = faces[0].height;
				int h2 = h;
				int w = faces[0].width;
				int w2 = w;
				rectangle(captured, Point(x, y), Point(x+w, y+h),
						Scalar(0, 255, 0), 2, 8, 0);

				if (h/w > 112/92) { // crop top and bottom
					h2 = (112*w)/92;
					y += (h-h2)/2;
				} else { // crop left and right
					w2 = (92*h)/112;
					x += (w-w2)/2;
				}

				int x2 = x + w2;
				int y2 = y + h2;

				rectangle(captured, Point(x, y), Point(x2, y2),
						Scalar(0, 255, 0), 2, 8, 0);

				imshow(WINDOW_NAME, captured);
				waitKey(30);

				Mat cropedFace = captured(Rect(x,y,w2,h2)).clone();

				Size size(92, 112);
				Mat resizedCropedFace;
				resize(cropedFace, resizedCropedFace, size);

				imshow(CROPED_WINDOW_NAME, resizedCropedFace);
				waitKey(30);
				return resizedCropedFace;
			} else {
				cout << "no face or multiple faces detected" << endl;
			}
		} catch (cv::Exception& e) {
			cerr << "Error: " << e.msg << endl;
		}
	}
}

void addTrainingDataForOneSubject(RaspiCam_Cv& camera, CascadeClassifier& face_cascade,
		vector<Mat>& trainingImages, vector<int>& trainingLabels, int subjectId, const char* name)
{
	string message = format("./speech.sh %s, bitte schaue mich an, so dass ich Dich kennen lernen kann", name);
	system(message.c_str());
	cout << "Adding training data for subject " << subjectId << endl;
	Mat captured;
	vector<Rect> faces;
	//for (int i = 0; i < 10; i++) {
	for (int i = 0; i < 10; i++) {
		Mat captured = captureSingleImage(camera, face_cascade);
		system("./speech.sh So kann ich Dich gut erkennen.");
		trainingImages.push_back(captured);
		trainingLabels.push_back(subjectId);
	}
	message = format("./speech.sh Das wars %s. Jetzt kenn ich Dich.", name);
	system(message.c_str());
	cout << "Done adding training data for subject " << subjectId << endl;
}

void trainFromCamera(RaspiCam_Cv& camera, CascadeClassifier& face_cascade, Ptr<FaceRecognizer>& model) {
	vector<Mat> trainingImages;
	vector<int> trainingLabels;
	int subjectCount = 0;
	string subjectName;
	cout << "Capturing training data from camera..." << endl;
	system("./speech.sh Wie viele Personen spielen mit?");
	cout << "Amount of subjects: ";
	cin >> subjectCount;
	for (int i = 0; i < subjectCount; i++) {
		string message = format("./speech.sh Tippe den Namen der %d. Person ein.", i+1);
		system(message.c_str());
		cout << "Capturing training data for subject " << i << ". Enter name:";
		cin >> subjectName;
		subjectNames.push_back(subjectName);
		addTrainingDataForOneSubject(camera, face_cascade, trainingImages, trainingLabels, i, subjectName.c_str());
	}
	cout << "Done capturing images. Starting training..." << endl;
	model->train(trainingImages, trainingLabels);
	cout << "Training completed." << endl;
	system("./speech So, jetzt kenne ich Euch. Es kann los gehen.");
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
			namedWindow(CROPED_WINDOW_NAME, WINDOW_AUTOSIZE);
			if (argc < 2) {
				trainFromCamera(camera, face_cascade, model);
			} else {
				string trainingDataCsvFile = string(argv[1]);
				trainFromCsv(trainingDataCsvFile, model);
			}

			while(1) {
				//std::vector<Rect> faces;
				Mat resizedCropedFace = captureSingleImage(camera, face_cascade);
				int predictedLabel = -1;
				double confidence = 0.0;

				//predictedLabel = model->predict(resizedCropedFace);
				model->predict(resizedCropedFace, predictedLabel, confidence);

				string result_message = format("Predicted label = %d, Confidence = %f.", predictedLabel, confidence);
				cout << result_message << endl;

				if (confidence < 2000.0) {
					string name = subjectNames[predictedLabel];
					cout << "Recognized subject " << name << endl;
					string message = format("./speech.sh Hallo %s! Wie geht es dir?", name.c_str());
					system(message.c_str());
				} else {
					cout << "Unrecogniced face." << endl;
					system("./speech.sh Hallo Fremder!");
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
