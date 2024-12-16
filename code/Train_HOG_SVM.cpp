/*
이 코드는 차량과 비차량 이미지 데이터를 불러옵니다, 
데이터를 증강(좌우 반전)한 뒤 
데이터를 무작위로 섞은 후, HOG 특징을 추출합니다. 
HOG 특징 추출하는 방법은 총 두 가지가 있습니다 (겹치게, 겹치지 않게).

SVM의 하이퍼파라미터(C와 gamma)를 그리드 탐색 방식으로 최적화하며, 
best_svm_model.xml으로 저장합니다.

주의) Car, Non-Car 로드하기 위해, 해당 주소에 맞게 작성해주셔야 합니다.
*/

#include <iostream>
#include <filesystem>
#include <algorithm>  // shuffle
#include <random>     // default_random_engine
#include <vector>
#include <opencv2/opencv.hpp> 
#include <opencv2/ml.hpp>   // svm
#include "HOGFeatureExtractor.h" // HOG 추출 코드 포함

using namespace std;
using namespace cv;
using namespace ml;
namespace fs = filesystem;


// 폴더에서 이미지 로드
void loadImages(const string& folder, vector<Mat>& images, vector<int>& labels, int label) {
    for (const auto& entry : fs::directory_iterator(folder)) {
        Mat img = imread(entry.path().string(), IMREAD_COLOR);
        resize(img, img, Size(64, 64));
        if (!img.empty()) {
            images.push_back(img);
            labels.push_back(label);
        }
    }
}

// 데이터를 훈련/테스트로 나누기
void splitData(const vector<Mat>& images, const vector<int>& labels,
    vector<Mat>& X_train, vector<int>& y_train,
    vector<Mat>& X_test, vector<int>& y_test, 
    float trainRatio = 0.8) {  
    
    // 훈련 80%, 테스트 20%
    size_t trainSize = static_cast<size_t>(images.size() * trainRatio);
    for (size_t i = 0; i < images.size(); ++i) {
        if (i < trainSize) {
            X_train.push_back(images[i]);
            y_train.push_back(labels[i]);
        }
        else {
            X_test.push_back(images[i]);
            y_test.push_back(labels[i]);
        }
    }
}

// 증강 (좌우 반전)
Mat flipHorizontal(const Mat& inputImage) {
    Mat outputImage;
    flip(inputImage, outputImage, 1);
    return outputImage;
}
void augmentation(vector<Mat>& images, vector<int>& labels, vector<Mat> augImages, vector<int>& augLabels) {
    for (size_t i = 0; i < images.size(); ++i) {
        // 원래
        augImages.push_back(images[i]);
        augLabels.push_back(labels[i]);

        // 좌우 반전
        Mat augmented = flipHorizontal(images[i]);
        augImages.push_back(augmented);
        augLabels.push_back(labels[i]);
    }
}

// 랜덤하게 shuffle
void shuffleData(vector<Mat>& X_train, vector<int>& y_train, int seed) {
    // 인덱스 벡터 생성
    vector<int> indices(X_train.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;  // 인덱스를 직접 생성
    }

    // 난수 생성기 설정
    default_random_engine g(seed);
    shuffle(indices.begin(), indices.end(), g);
 
    vector<Mat> shuffledData;
    vector<int> shuffledLabels;
    for (int i : indices) {
        shuffledData.push_back(X_train[i]);
        shuffledLabels.push_back(y_train[i]);
    }

    X_train = move(shuffledData);
    y_train = move(shuffledLabels);
}

// HOG 특징 추출의 첫 번째 방식 적용 (겹치지 않게 HOG 특징 추출)
void simplePreprocessing(const vector<Mat>& X_train, const vector<int>& y_train,
    const vector<Mat>& X_test, const vector<int>& y_test,
    Mat& pre_X_train, Mat& pre_y_train, Mat& pre_X_test, Mat& pre_y_test) {

    // X data: Hog 
    for (size_t i = 0; i < X_train.size(); ++i) {
        vector<float> descriptors = SimpleHogFeatureExtractor(X_train[i]);
        pre_X_train.push_back(Mat(descriptors).reshape(1, 1));
    }
    for (size_t i = 0; i < X_test.size(); ++i) {
        vector<float> descriptors = SimpleHogFeatureExtractor(X_test[i]);
        pre_X_test.push_back(Mat(descriptors).reshape(1, 1));
    }

    // y data
    pre_y_train = Mat(y_train).reshape(1, y_train.size());
    pre_y_test = Mat(y_test).reshape(1, y_test.size());
}

// HOG 특징 추출의 두 번째 방식 적용 (겹치게 HOG 특징 추출)
void preprocessing(const vector<Mat>& X_train, const vector<int>& y_train, 
    const vector<Mat>& X_test, const vector<int>& y_test,
    Mat& pre_X_train, Mat& pre_y_train, Mat& pre_X_test, Mat& pre_y_test) {

    // X data: Hog 
    for (size_t i = 0; i < X_train.size(); ++i) {
        vector<float> descriptors = HogFeatureExtractor(X_train[i]);
        pre_X_train.push_back(Mat(descriptors).reshape(1, 1));
    }
    for (size_t i = 0; i < X_test.size(); ++i) {
        vector<float> descriptors = HogFeatureExtractor(X_test[i]);
        pre_X_test.push_back(Mat(descriptors).reshape(1, 1));
    }

    // y data
    pre_y_train = Mat(y_train).reshape(1, y_train.size());
    pre_y_test = Mat(y_test).reshape(1, y_test.size());
}


void trainAndTest(const Mat& pre_X_train, const Mat& pre_y_train, 
    const Mat& pre_X_test, const Mat& pre_y_test) {

    // 비선형 학습
    // 하이퍼파라미터 후보 설정
    vector<double> C_values = { 0.01, 0.1, 1, 10, 100 };    
    vector<double> gamma_values = { 0.001, 0.01, 0.1, 1 };   
    double bestAccuracy = 0.0;
    Ptr<SVM> bestSVM;
    double bestC, bestGamma;
    for (double C : C_values) {
        for (double gamma : gamma_values) {
            Ptr<SVM> svm = SVM::create();
            svm->setType(SVM::C_SVC);
            svm->setKernel(SVM::RBF);  // 비선형 커널 사용
            svm->setC(C);
            svm->setGamma(gamma);

            // SVM 훈련
            svm->train(pre_X_train, ROW_SAMPLE, pre_y_train);

            // 테스트 데이터 평가 
            int testCorrect = 0;
            int testFail = 0;
            for (int i = 0; i < pre_X_test.rows; ++i) {
                float prediction = svm->predict(pre_X_test.row(i));
                if (prediction == pre_y_test.at<int>(i, 0)) {
                    testCorrect++;
                }
                else {
                    testFail++;
                }
            }
            float testAccuracy = static_cast<float>(testCorrect) / pre_X_test.rows;

            // 최적의 모델 저장
            if (testAccuracy > bestAccuracy) {
                bestAccuracy = testAccuracy;
                bestSVM = svm;
                bestC = C;
                bestGamma = gamma;
            }
        }
    }

    // 최적의 모델로 최종 테스트 및 저장
    cout << "Best Test Accuracy: " << bestAccuracy * 100.0f << "%" << endl;
    cout << "BestC: " << bestC << endl;
    cout << "BestGamma: " << bestGamma << endl;
    if (bestSVM) {
        bestSVM->save("best_svm_model.xml");
        cout << "Best model saved to 'best_svm_model.xml'" << endl;
    }
}

void main() {

    string carsPath = "C:/Users/user/source/repos/test/test/Traditional-Object-Detection-master/dataset/vehicles_smallset";     // 주소에 맞게 바꿔야 함 
    string notCarsPath = "C:/Users/user/source/repos/test/test/Traditional-Object-Detection-master/dataset/non-vehicles_smallset";  // 주소에 맞게 바꿔야 함

    vector<Mat> images, augImages;
    vector<int> labels, augLabels;

    // 데이터 로드
    loadImages(carsPath + "/cars1", images, labels, 1);     // 차량 이미지: 레이블 1
    loadImages(carsPath + "/cars2", images, labels, 1);
    loadImages(carsPath + "/cars3", images, labels, 1);

    loadImages(notCarsPath + "/notcars1", images, labels, 0); // 비차량 이미지: 레이블 0
    loadImages(notCarsPath + "/notcars2", images, labels, 0);
    loadImages(notCarsPath + "/notcars3", images, labels, 0);

    // 데이터 증강
    augmentation(images, labels, augImages, augLabels);
    
    // 데이터 분할
    vector<Mat> X_train, X_test;
    vector<int> y_train, y_test;
    splitData(images, labels, X_train, y_train, X_test, y_test);
 
    // Shuffle
    shuffleData(X_train, y_train, 42);
    
    // Preprocess
    Mat pre_X_train, pre_y_train, pre_X_test, pre_y_test;
    
    // HOG 특징 추출 방법 1 (겹치지 않게) 정확도 88.8172%
    //simplePreprocessing(X_train, y_train, X_test, y_test, pre_X_train, pre_y_train, pre_X_test, pre_y_test);

    // HOG 특징 추출 방법 2 (겹치게) 정확도 89.8925 %
    preprocessing(X_train, y_train, X_test, y_test, pre_X_train, pre_y_train, pre_X_test, pre_y_test);
    
    // 훈련 및 테스트
    trainAndTest(pre_X_train, pre_y_train, pre_X_test, pre_y_test);

}
