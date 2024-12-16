/*
�ڵ��� �����󿡼� �ڵ������� Detection �ϴ� �ڵ��Դϴ�.
����, (best_svm_model.xml)�� �����ؾ� ������ �˴ϴ�.
*/

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include "HOGFeatureExtractor.h" // �ۼ��� HOG ���� �ڵ� 

using namespace std;
using namespace cv;
using namespace ml;
namespace fs = std::filesystem;


vector<Mat> generateImagePyramid(const Mat& image, float scaleFactor = 0.8, int minSize = 120, int maxLevels = 5) {
    /*
    �̹��� �Ƕ�̵� ���� �Լ�
    
    ���: 
    [�߶��� ����: (1640, 830),                   --- ���� ���� ã���
    ���� 0.8x: (1312, 664 ), 
    ���� 0.8^2x: (1049.6, 531.2),                --- �߰� ���� ã���
    ���� 0.8^3x: (839.68, 424.96)]               
    ���� 0.8^4x: (671.744, 339.968)]             --- ū ���� ã���
    */

    vector<Mat> pyramid;
    Mat current = image.clone();
    int level = 0;
    while (current.rows >= minSize && current.cols >= minSize && level < maxLevels) {
        pyramid.push_back(current);
        resize(current, current, Size(), scaleFactor, scaleFactor);
        level++;
    }
    return pyramid; 
}

vector<Rect> slidingWindow(const Mat& image, int minWinWidth = 136, int minWinHeight = 110) {
    /*
    �����̵� ������ �Լ�    
    �Էµ� ���� ������ �����̵� ��������� ���͸� �����մϴ�.
    */
    vector<Rect> windows;
    int y_stride = minWinHeight / 4;
    int x_stride = minWinWidth / 4;     
    for (int y = 0; y <= image.rows - minWinHeight; y += y_stride) {
        for (int x = 0; x <= image.cols - minWinWidth; x += x_stride) {
            windows.push_back(Rect(x, y, minWinWidth, minWinHeight));
        }
    }
    return windows;
}


vector<Rect> detectObjects(Ptr<SVM> svm, const vector<Mat>& pyramid, vector<float>&scores, const Rect& roiRect) {
    /*
    ��ü Ž�� �Լ�
    �� �Ƕ�̵� ������ ù ��° ������ �κ��� Ȯ���� �˰� ������, 
    �ؿ� �ִ� �ּ��� ���ָ� �˴ϴ�.
    */
    vector<Rect> detections;
    for (size_t i = 0; i < pyramid.size(); i++) {
        vector<Rect> windows = slidingWindow(pyramid[i]);
        int first_window = 0;   // �� �Ƕ�̵��� ù�� ° ������ Ȯ�ο�
        for (const Rect& window : windows) {
            Mat roi = pyramid[i](window); // �����̵� ������ ���� ���� (120 x 150)
            /*
            // ���� ������ 
            imshow("current_window", roi);
            */
            
            resize(roi, roi, Size(64, 64)); // �н� ������ ũ���� 64x64�� ��������
            vector<float> descriptors = HogFeatureExtractor(roi); // HOG Ư¡ ����
            Mat featureVector = Mat(descriptors).reshape(1, 1);
            float rawScore = svm->predict(featureVector, noArray(), StatModel::RAW_OUTPUT);
            float probability = 1.0 / (1.0 + exp(-rawScore)); // Sigmoid �Լ��� Ȯ�� ���
            probability = 1 - probability;  // �Ųٷ� �Ǿ����� 
            
            /*
            // �� �Ƕ�̵��� ù ��° �������� Ȯ���� ���
            if (first_window == 0) {
                cout << "probability: " << probability << endl;
                waitKey(0);
            }
            first_window += 1;
            */
            
            if (probability > 0.5) { // �ڵ������ �Ǵ�
                // ���� �����Ϸ� ��ǥ ��ȯ
                float scale = pow(0.8, static_cast<int>(i)); // ������ ����
                Rect originalWindow = Rect(                  // �ڵ��� ����
                    static_cast<int>(window.x / scale),
                    static_cast<int>(window.y / scale),
                    static_cast<int>(window.width / scale),
                    static_cast<int>(window.height / scale)
                );
                
                // crop�� �κ� ���
                originalWindow.x += roiRect.x;
                originalWindow.y += roiRect.y;
                detections.push_back(originalWindow);
                scores.push_back(probability); // Ȯ�� ���� score�� �߰�
            }
        }
    }
    return detections;
}

vector<Rect> applyNMS(const vector<Rect>& detections, const vector<float>& scores, float nmsThreshold) {
    /*
    ���ִ� ����(NMS) ���� �Լ�
    �ߺ��� ��ü Ž���� ���̱� ����
    */
    vector<int> indices;
    vector<Rect> finalDetections;
    // OpenCV dnn::NMSBoxes �Լ� ����Ͽ� NMS ����
    dnn::NMSBoxes(detections, scores, 0, nmsThreshold, indices);   

    // ���� ���õ� �ڽ����� ��� ���Ϳ� �߰�
    for (int idx : indices) {
        finalDetections.push_back(detections[idx]);
    }
    return finalDetections;
}


int main() {
    // �̹��� �ε�

    //Video I/O --------------------------------
    VideoCapture capture("car_video2.mp4");

    Mat image;
    namedWindow("video");
    while (1) {
        capture >> image;
        if (image.empty())
            break;
        
        // �н��� SVM �� �ε�
        Ptr<SVM> svm = SVM::load("best_svm_model.xml");
        if (svm.empty()) {
            cerr << "Error loading SVM model!" << endl;
            return -1;
        }
        
        //Crop Image
        // ���� ũ��(1920, 1080) - > ũ�� (1640, 830)
        Rect roiRect(280, 250, 1640, 830);  // ~������ 
        Mat croppedImage = image(roiRect);

        // �̹��� �Ƕ�̵� ����
        vector<Mat> pyramid = generateImagePyramid(croppedImage);

        // ��ü Ž��
        vector<float> scores;
        vector<Rect> detections = detectObjects(svm, pyramid, scores, roiRect);
        
        // NMS ����
        vector<Rect> finalDetections = applyNMS(detections, scores, 0.06); 

        // ��� �ð�ȭ
        for (const Rect& box : finalDetections) { 
            rectangle(image, box, Scalar(0, 255, 0), 2); // Ž�� ��� �׸���
        }
        imshow("video", image);

        //waitKey(0);   // �� �����Ӿ� ����ʹٸ�
        if (waitKey(10) > 0)
            break;

    }
    
    return 0;
}
