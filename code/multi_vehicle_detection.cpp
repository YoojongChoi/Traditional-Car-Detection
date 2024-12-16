/*
자동차 동영상에서 자동차들을 Detection 하는 코드입니다.
주의, (best_svm_model.xml)이 존재해야 실행이 됩니다.
*/

#include <iostream>
#include <filesystem> 
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include "HOGFeatureExtractor.h" // 작성한 HOG 추출 코드 

using namespace std;
using namespace cv;
using namespace ml;
namespace fs = std::filesystem;


vector<Mat> generateImagePyramid(const Mat& image, float scaleFactor = 0.8, int minSize = 120, int maxLevels = 5) {
    /*
    이미지 피라미드 생성 함수
    
    출력: 
    [잘라진 원본: (1640, 830),                   --- 작은 차들 찾기용
    원본 0.8x: (1312, 664 ), 
    원본 0.8^2x: (1049.6, 531.2),                --- 중간 차들 찾기용
    원본 0.8^3x: (839.68, 424.96)]               
    원본 0.8^4x: (671.744, 339.968)]             --- 큰 차들 찾기용
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
    슬라이딩 윈도우 함수    
    입력된 영상 위에서 슬라이딩 윈도우들의 벡터를 생성합니다.
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
    객체 탐지 함수
    각 피라미드 영상의 첫 번째 윈도우 부분의 확률을 알고 싶으면, 
    밑에 있는 주석을 없애면 됩니다.
    */
    vector<Rect> detections;
    for (size_t i = 0; i < pyramid.size(); i++) {
        vector<Rect> windows = slidingWindow(pyramid[i]);
        int first_window = 0;   // 각 피라미드의 첫번 째 윈도우 확인용
        for (const Rect& window : windows) {
            Mat roi = pyramid[i](window); // 슬라이딩 윈도우 영역 추출 (120 x 150)
            /*
            // 현재 윈도우 
            imshow("current_window", roi);
            */
            
            resize(roi, roi, Size(64, 64)); // 학습 데이터 크기인 64x64로 리사이즈
            vector<float> descriptors = HogFeatureExtractor(roi); // HOG 특징 추출
            Mat featureVector = Mat(descriptors).reshape(1, 1);
            float rawScore = svm->predict(featureVector, noArray(), StatModel::RAW_OUTPUT);
            float probability = 1.0 / (1.0 + exp(-rawScore)); // Sigmoid 함수로 확률 계산
            probability = 1 - probability;  // 거꾸로 되어있음 
            
            /*
            // 각 피라미드의 첫 번째 윈도우의 확률을 출력
            if (first_window == 0) {
                cout << "probability: " << probability << endl;
                waitKey(0);
            }
            first_window += 1;
            */
            
            if (probability > 0.5) { // 자동차라고 판단
                // 원본 스케일로 좌표 변환
                float scale = pow(0.8, static_cast<int>(i)); // 스케일 비율
                Rect originalWindow = Rect(                  // 자동차 구역
                    static_cast<int>(window.x / scale),
                    static_cast<int>(window.y / scale),
                    static_cast<int>(window.width / scale),
                    static_cast<int>(window.height / scale)
                );
                
                // crop한 부분 고려
                originalWindow.x += roiRect.x;
                originalWindow.y += roiRect.y;
                detections.push_back(originalWindow);
                scores.push_back(probability); // 확률 값을 score에 추가
            }
        }
    }
    return detections;
}

vector<Rect> applyNMS(const vector<Rect>& detections, const vector<float>& scores, float nmsThreshold) {
    /*
    비최대 억제(NMS) 적용 함수
    중복된 객체 탐지를 줄이기 위함
    */
    vector<int> indices;
    vector<Rect> finalDetections;
    // OpenCV dnn::NMSBoxes 함수 사용하여 NMS 적용
    dnn::NMSBoxes(detections, scores, 0, nmsThreshold, indices);   

    // 최종 선택된 박스들을 결과 벡터에 추가
    for (int idx : indices) {
        finalDetections.push_back(detections[idx]);
    }
    return finalDetections;
}


int main() {
    // 이미지 로드

    //Video I/O --------------------------------
    VideoCapture capture("car_video2.mp4");

    Mat image;
    namedWindow("video");
    while (1) {
        capture >> image;
        if (image.empty())
            break;
        
        // 학습된 SVM 모델 로드
        Ptr<SVM> svm = SVM::load("best_svm_model.xml");
        if (svm.empty()) {
            cerr << "Error loading SVM model!" << endl;
            return -1;
        }
        
        //Crop Image
        // 원본 크기(1920, 1080) - > 크기 (1640, 830)
        Rect roiRect(280, 250, 1640, 830);  // ~끝까지 
        Mat croppedImage = image(roiRect);

        // 이미지 피라미드 생성
        vector<Mat> pyramid = generateImagePyramid(croppedImage);

        // 객체 탐지
        vector<float> scores;
        vector<Rect> detections = detectObjects(svm, pyramid, scores, roiRect);
        
        // NMS 적용
        vector<Rect> finalDetections = applyNMS(detections, scores, 0.06); 

        // 결과 시각화
        for (const Rect& box : finalDetections) { 
            rectangle(image, box, Scalar(0, 255, 0), 2); // 탐지 결과 그리기
        }
        imshow("video", image);

        //waitKey(0);   // 한 프레임씩 보고싶다면
        if (waitKey(10) > 0)
            break;

    }
    
    return 0;
}
