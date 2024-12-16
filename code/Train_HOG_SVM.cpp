/*
�� �ڵ�� ������ ������ �̹��� �����͸� �ҷ��ɴϴ�, 
�����͸� ����(�¿� ����)�� �� 
�����͸� �������� ���� ��, HOG Ư¡�� �����մϴ�. 
HOG Ư¡ �����ϴ� ����� �� �� ������ �ֽ��ϴ� (��ġ��, ��ġ�� �ʰ�).

SVM�� �������Ķ����(C�� gamma)�� �׸��� Ž�� ������� ����ȭ�ϸ�, 
best_svm_model.xml���� �����մϴ�.

����) Car, Non-Car �ε��ϱ� ����, �ش� �ּҿ� �°� �ۼ����ּž� �մϴ�.
*/

#include <iostream>
#include <filesystem>
#include <algorithm>  // shuffle
#include <random>     // default_random_engine
#include <vector>
#include <opencv2/opencv.hpp> 
#include <opencv2/ml.hpp>   // svm
#include "HOGFeatureExtractor.h" // HOG ���� �ڵ� ����

using namespace std;
using namespace cv;
using namespace ml;
namespace fs = filesystem;


// �������� �̹��� �ε�
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

// �����͸� �Ʒ�/�׽�Ʈ�� ������
void splitData(const vector<Mat>& images, const vector<int>& labels,
    vector<Mat>& X_train, vector<int>& y_train,
    vector<Mat>& X_test, vector<int>& y_test, 
    float trainRatio = 0.8) {  
    
    // �Ʒ� 80%, �׽�Ʈ 20%
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

// ���� (�¿� ����)
Mat flipHorizontal(const Mat& inputImage) {
    Mat outputImage;
    flip(inputImage, outputImage, 1);
    return outputImage;
}
void augmentation(vector<Mat>& images, vector<int>& labels, vector<Mat> augImages, vector<int>& augLabels) {
    for (size_t i = 0; i < images.size(); ++i) {
        // ����
        augImages.push_back(images[i]);
        augLabels.push_back(labels[i]);

        // �¿� ����
        Mat augmented = flipHorizontal(images[i]);
        augImages.push_back(augmented);
        augLabels.push_back(labels[i]);
    }
}

// �����ϰ� shuffle
void shuffleData(vector<Mat>& X_train, vector<int>& y_train, int seed) {
    // �ε��� ���� ����
    vector<int> indices(X_train.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;  // �ε����� ���� ����
    }

    // ���� ������ ����
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

// HOG Ư¡ ������ ù ��° ��� ���� (��ġ�� �ʰ� HOG Ư¡ ����)
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

// HOG Ư¡ ������ �� ��° ��� ���� (��ġ�� HOG Ư¡ ����)
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

    // ���� �н�
    // �������Ķ���� �ĺ� ����
    vector<double> C_values = { 0.01, 0.1, 1, 10, 100 };    
    vector<double> gamma_values = { 0.001, 0.01, 0.1, 1 };   
    double bestAccuracy = 0.0;
    Ptr<SVM> bestSVM;
    double bestC, bestGamma;
    for (double C : C_values) {
        for (double gamma : gamma_values) {
            Ptr<SVM> svm = SVM::create();
            svm->setType(SVM::C_SVC);
            svm->setKernel(SVM::RBF);  // ���� Ŀ�� ���
            svm->setC(C);
            svm->setGamma(gamma);

            // SVM �Ʒ�
            svm->train(pre_X_train, ROW_SAMPLE, pre_y_train);

            // �׽�Ʈ ������ �� 
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

            // ������ �� ����
            if (testAccuracy > bestAccuracy) {
                bestAccuracy = testAccuracy;
                bestSVM = svm;
                bestC = C;
                bestGamma = gamma;
            }
        }
    }

    // ������ �𵨷� ���� �׽�Ʈ �� ����
    cout << "Best Test Accuracy: " << bestAccuracy * 100.0f << "%" << endl;
    cout << "BestC: " << bestC << endl;
    cout << "BestGamma: " << bestGamma << endl;
    if (bestSVM) {
        bestSVM->save("best_svm_model.xml");
        cout << "Best model saved to 'best_svm_model.xml'" << endl;
    }
}

void main() {

    string carsPath = "C:/Users/user/source/repos/test/test/Traditional-Object-Detection-master/dataset/vehicles_smallset";     // �ּҿ� �°� �ٲ�� �� 
    string notCarsPath = "C:/Users/user/source/repos/test/test/Traditional-Object-Detection-master/dataset/non-vehicles_smallset";  // �ּҿ� �°� �ٲ�� ��

    vector<Mat> images, augImages;
    vector<int> labels, augLabels;

    // ������ �ε�
    loadImages(carsPath + "/cars1", images, labels, 1);     // ���� �̹���: ���̺� 1
    loadImages(carsPath + "/cars2", images, labels, 1);
    loadImages(carsPath + "/cars3", images, labels, 1);

    loadImages(notCarsPath + "/notcars1", images, labels, 0); // ������ �̹���: ���̺� 0
    loadImages(notCarsPath + "/notcars2", images, labels, 0);
    loadImages(notCarsPath + "/notcars3", images, labels, 0);

    // ������ ����
    augmentation(images, labels, augImages, augLabels);
    
    // ������ ����
    vector<Mat> X_train, X_test;
    vector<int> y_train, y_test;
    splitData(images, labels, X_train, y_train, X_test, y_test);
 
    // Shuffle
    shuffleData(X_train, y_train, 42);
    
    // Preprocess
    Mat pre_X_train, pre_y_train, pre_X_test, pre_y_test;
    
    // HOG Ư¡ ���� ��� 1 (��ġ�� �ʰ�) ��Ȯ�� 88.8172%
    //simplePreprocessing(X_train, y_train, X_test, y_test, pre_X_train, pre_y_train, pre_X_test, pre_y_test);

    // HOG Ư¡ ���� ��� 2 (��ġ��) ��Ȯ�� 89.8925 %
    preprocessing(X_train, y_train, X_test, y_test, pre_X_train, pre_y_train, pre_X_test, pre_y_test);
    
    // �Ʒ� �� �׽�Ʈ
    trainAndTest(pre_X_train, pre_y_train, pre_X_test, pre_y_test);

}
