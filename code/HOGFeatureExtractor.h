/*
OpenCV 함수없이 HOG 특징 함수를 추출하기 위한 코드입니다.
추출된 특징을 Visualize 하고 싶다면 주석을 지우면 됩니다.

HOG 특징을 추출하기 위한 방법 2가지

방법1) 겹치지 않게: Class MySimpleHOGDescriptor (정확도 88.8172%)
방법2) 겹치게: Class MyHOGDescriptor (정확도 89.8925 %)
*/

#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// 1차원 배열 할당 함수
unsigned char* MatToArray(Mat frame, int rows, int cols) {
    unsigned char* array = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
    for (int h = 0; h < rows; h++) {
        for (int w = 0; w < cols; w++) {
            array[h * cols + w] = frame.at<uchar>(h, w);
        }
    }
    return array;
}

template <typename T>
Mat ArrayToMat(const T* src, int rows, int cols) {
    int type = (is_same<T, float>::value) ? CV_32F : CV_8U;
    Mat mat(rows, cols, type);
    for (int h = 0; h < rows; h++) {
        for (int w = 0; w < cols; w++) {
            mat.at<T>(h, w) = src[h * cols + w];
        }
    }
    return mat;
}

// 패딩
unsigned char* Padding(const unsigned char* src, int rows, int cols, int nFilterSize) {
    int nPadSize = (int)nFilterSize / 2;
    int paddedRows = rows + 2 * nPadSize;
    int paddedCols = cols + 2 * nPadSize;
    unsigned char* dst = (unsigned char*)malloc(paddedRows * paddedCols * sizeof(unsigned char));

    // 가운데 영역
    for (int h = 0; h < rows; h++) {
        for (int w = 0; w < cols; w++) {
            dst[(h + nPadSize) * paddedCols + (w + nPadSize)] = src[h * cols + w];
        }
    }
    // 위, 아래 패딩
    for (int w = 0; w < cols; w++) {
        for (int p = 0; p < nPadSize; p++) {
            dst[p * paddedCols + (w + nPadSize)] = src[w];  // 위쪽
            dst[(paddedRows - nPadSize + p) * paddedCols + (w + nPadSize)] = src[(rows - 1) * cols + w];  // 아래쪽
        }
    }
    // 왼쪽, 오른쪽 패딩
    for (int h = 0; h < rows; h++) {
        for (int p = 0; p < nPadSize; p++) {
            dst[(h + nPadSize) * paddedCols + p] = src[h * cols];  // 왼쪽
            dst[(h + nPadSize) * paddedCols + (paddedCols - nPadSize + p)] = src[h * cols + (cols - 1)];  // 오른쪽
        }
    }
    // 모서리 패딩
    for (int p = 0; p < nPadSize; p++) {
        for (int q = 0; q < nPadSize; q++) {
            dst[p * paddedCols + q] = src[0];  // 왼쪽 위
            dst[p * paddedCols + (paddedCols - nPadSize + q)] = src[cols - 1];  // 오른쪽 위
            dst[(paddedRows - nPadSize + p) * paddedCols + q] = src[(rows - 1) * cols];  // 왼쪽 아래
            dst[(paddedRows - nPadSize + p) * paddedCols + (paddedCols - nPadSize + q)] = src[(rows - 1) * cols + (cols - 1)];  // 오른쪽 아래
        }
    }
    return dst;
}

// Convolution
unsigned char* Convolution(const unsigned char* src, int rows, int cols, int nFilterSize, const int kernel[3][3]) {
    int nPadSize = (int)nFilterSize / 2;
    int paddedRows = rows + 2 * nPadSize;
    int paddedCols = cols + 2 * nPadSize;
    unsigned char* dst = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));

    // 패딩된 입력 배열 생성
    unsigned char* paddedSrc = Padding(src, rows, cols, nFilterSize);

    // Convolution 연산
    for (int h = nPadSize; h < paddedRows - nPadSize; h++) {
        for (int w = nPadSize; w < paddedCols - nPadSize; w++) {
            float sum = 0.0f;
            for (int kh = -nPadSize; kh <= nPadSize; kh++) {
                for (int kw = -nPadSize; kw <= nPadSize; kw++) {
                    int paddedIndex = (h + kh) * paddedCols + (w + kw);
                    sum += paddedSrc[paddedIndex] * kernel[kh + nPadSize][kw + nPadSize];
                }
            }

            sum = max(0.0f, min(255.0f, sum));
            dst[(h - nPadSize) * cols + (w - nPadSize)] = static_cast<unsigned char>(sum);
        }
    }
    free(paddedSrc);
    return dst;
}

// Sobel Edge 연산
pair<unsigned char*, unsigned char*> SobelEdge(const unsigned char* src, int rows, int cols) {
    // Sobel 마스크
    int sobel_H[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int sobel_V[3][3] = { { 1, 2, 1}, { 0, 0, 0}, {-1, -2, -1} };

    // Convolution
    unsigned char* frame_h = Convolution(src, rows, cols, 3, sobel_H);
    unsigned char* frame_v = Convolution(src, rows, cols, 3, sobel_V);

    return make_pair(frame_h, frame_v);
}

// Magnitude & Orientation 추출
pair<float*, float*> MagnitudeOrientation(const unsigned char* frame_h, const unsigned char* frame_v, int rows, int cols) {
    // 기울기 벡터를 이루는 요소들
    float* magnitude = (float*)malloc(rows * cols * sizeof(float));     // 크기
    float* orientation = (float*)malloc(rows * cols * sizeof(float));   // 방향

    for (int h = 0; h < rows; h++) {
        for (int w = 0; w < cols; w++) {
            magnitude[h * cols + w] = sqrt(frame_h[h * cols + w] * frame_h[h * cols + w] + frame_v[h * cols + w] * frame_v[h * cols + w]);
            orientation[h * cols + w] = atan2(frame_v[h * cols + w], frame_h[h * cols + w]) * 180.0 / CV_PI;
            if (orientation[h * cols + w] < 0)
                orientation[h * cols + w] += 360;
        }
    }
    return make_pair(magnitude, orientation);
}

// Visualize Magnitude
void visualizeMagnitude(float* magnitude, int rows, int cols) {
    Mat magnitude_img(rows, cols, CV_8UC1);
    for (int h = 0; h < rows; h++) {
        for (int w = 0; w < cols; w++) {
            magnitude_img.at<uchar>(h, w) = static_cast<uchar>(magnitude[h * cols + w]);
        }
    }
    imshow("Mag", magnitude_img);
}

// Visualize Orientation
void visualizeOrientation(float* orientation, int rows, int cols) {
    Mat direction_img = Mat::zeros(rows, cols, CV_8UC1);

    for (int h = 0; h < rows; h += 10) {  // 10 픽셀마다 표시
        for (int w = 0; w < cols; w += 10) {
            float angle = orientation[h * cols + w] * CV_PI / 180.0;  // 라디안 변환
            int mag = 5;
            // 시작점
            Point start(w, h);
            // 화살표 끝점: 크기와 방향을 고려해 계산
            Point end(w + static_cast<int>(mag * cos(angle)), h - static_cast<int>(mag * sin(angle)));

            // 방향을 화살표로 표시 (색상은 흰색으로 설정)
            arrowedLine(direction_img, start, end, Scalar(255), 1);
        }
    }
    imshow("Orientation", direction_img);
}

// 해당 자리에 Magnitude와 Orientation을 같이 Visualize 
void visualizeDirectionInGrayscale(float* magnitude, float* orientation, int rows, int cols) {
    // 흑백 이미지를 위한 배경 초기화 (검정색)
    Mat direction_img = Mat::zeros(rows, cols, CV_8UC1);

    for (int h = 0; h < rows; h += 10) {  // 10 픽셀마다 표시
        for (int w = 0; w < cols; w += 10) {
            float mag = magnitude[h * cols + w];
            float angle = orientation[h * cols + w] * CV_PI / 180.0;  // 라디안 변환

            // 시작점
            Point start(w, h);
            // 화살표 끝점: 크기와 방향을 고려해 계산
            Point end(w + static_cast<int>(mag * cos(angle)),
                h - static_cast<int>(mag * sin(angle)));

            // 방향을 화살표로 표시 (색상은 흰색으로 설정)
            arrowedLine(direction_img, start, end, Scalar(255), 1);
        }
    }
    imshow("Magnitude & Orientation", direction_img);
}

// Visualize Histogram
void visualizeHistogram(const std::vector<float>& histogram, int binCount = 9) {
    int hist_w = 512; 
    int hist_h = 400; 
    int bin_w = cvRound((double)hist_w / binCount);

    Mat histImage(hist_h + 50, hist_w, CV_8UC1, Scalar(255));
    for (int i = 0; i < binCount; ++i) {
        int bin_height = cvRound(histogram[i] * hist_h); 

        rectangle(histImage,
            Point(i * bin_w, hist_h),
            Point((i + 1) * bin_w, hist_h - bin_height),
            Scalar(0),
            FILLED);

        // 각 구역 bin 표시
        string binLabel = "bin: " + to_string(i + 1); 
        int text_x = i * bin_w + bin_w / 4; 
        int text_y = hist_h + 30; 

        putText(histImage, binLabel, Point(text_x, text_y),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0), 1);
    }

    imshow("Cell Histogram", histImage);
}

// HOG 특징 추출하기 위한 방법1: 겹치지 않게
class MySimpleHOGDescriptor {
public:
    MySimpleHOGDescriptor(Size cellSize, int nbins)
        : cellSize(cellSize), nbins(nbins) {}

    void compute(const Mat& colorFrame, std::vector<float>& descriptors) {
        // RGB to Gray
        Mat grayFrame;
        int rows = colorFrame.rows;
        int cols = colorFrame.cols;
        cvtColor(colorFrame, grayFrame, COLOR_BGR2GRAY);

        // Mat to Array
        unsigned char* grayArray = MatToArray(grayFrame, rows, cols);

        // Find gradient
        pair<unsigned char*, unsigned char*> edge = SobelEdge(grayArray, rows, cols);   // frame_h, frame_v

        // Find Magnitude and Orientation
        pair<float*, float*> magOri = MagnitudeOrientation(edge.first, edge.second, rows, cols);   // magnitude, orientation
        
        /*
        // Visualize Magnitude and Orientation
        
        visualizeMagnitude(magOri.first, rows, cols);   // Magnitude
        visualizeOrientation(magOri.second, rows, cols);    // Orientation
        visualizeDirectionInGrayscale(magOri.first, magOri.second, rows, cols); // Magnitude & Orientation
        */

        for (int i = 0; i < rows; i += cellSize.height) {
            for (int j = 0; j < cols; j += cellSize.width) {
                vector<float> cellHist(nbins, 0);

                // Calculate histogram for each cell
                for (int y = i; y < i + cellSize.height && y < rows; ++y) {
                    for (int x = j; x < j + cellSize.width && x < cols; ++x) {
                        float mag = magOri.first[y * cols + x];
                        float ang = magOri.second[y * cols + x];

                        int bin = static_cast<int>(nbins * (ang >= 180 ? ang - 180 : ang) / 180.0) % nbins;
                        cellHist[bin] += mag;
                    }
                }

                // Normalize the cell histogram (L2 norm)
                float cellSum = 0.0;
                for (float val : cellHist) {
                    cellSum += val * val;
                }
                cellSum = std::sqrt(cellSum) + 1e-6;
                for (float& val : cellHist) {
                    val /= cellSum;
                }
                
                /*
                //Visualize Histogram
                
                visualizeHistogram(cellHist);
                waitKey(0);
                */
                
                // Add the cell histogram to the final descriptor vector
                descriptors.insert(descriptors.end(), cellHist.begin(), cellHist.end());
            }
        }

        delete[] grayArray;
        delete[] edge.first;
        delete[] edge.second;
        delete[] magOri.first;
        delete[] magOri.second;
    }

private:
    Size cellSize;
    int nbins;
};
vector<float> SimpleHogFeatureExtractor(Mat colorFrame) {
    // 방법1 (겹치지 않게): 정확도 88.8172%
    Size cellSize(8, 8);
    int nbins = 9;

    // Create Simple HOG descriptor
    MySimpleHOGDescriptor hog(cellSize, nbins);

    // Compute HOG features
    vector<float> descriptors;
    hog.compute(colorFrame, descriptors);
    return descriptors;
}

// HOG 특징 추출하기 위한 방법2: 겹치게
class MyHOGDescriptor {
public:
    MyHOGDescriptor(Size winSize, Size blockSize, Size cellSize, Size winStride, Size blockStride,  int nbins)
        : winSize(winSize), blockSize(blockSize), cellSize(cellSize), winStride(winStride), blockStride(blockStride), nbins(nbins) {}

    void compute(const Mat& colorFrame, std::vector<float>& descriptors) {
        // RGB to Gray
        Mat grayFrame;
        int rows = colorFrame.rows; // 400
        int cols = colorFrame.cols; // 600
        cvtColor(colorFrame, grayFrame, COLOR_BGR2GRAY);

        // Mat to Array
        unsigned char* grayArray = MatToArray(grayFrame, rows, cols);

        // Find gradient
        pair<unsigned char*, unsigned char*> edge = SobelEdge(grayArray, rows, cols);   // frame_h, frame_v

        // Find Magnitude and Orientation
        pair<float*, float*> magOri = MagnitudeOrientation(edge.first, edge.second, rows, cols);   // magnitude, orientation

        /*
        // Visualize Magnitude and Orientation

        visualizeMagnitude(magOri.first, rows, cols);   // Magnitude
        visualizeOrientation(magOri.second, rows, cols);    // Orientation
        visualizeDirectionInGrayscale(magOri.first, magOri.second, rows, cols); // Magnitude & Orientation
        */

        for (int i = 0; i <= rows - winSize.height; i += winStride.height) {
            for (int j = 0; j <= cols - winSize.width; j += winStride.width) {
                vector<float> windowHist;

                // 각 윈도우 내에서 blockStride만큼 블록 슬라이딩
                // blcokSize > blockStride 
                for (int bi = 0; bi <= winSize.height - blockSize.height; bi += blockStride.height) {
                    for (int bj = 0; bj <= winSize.width - blockSize.width; bj += blockStride.width) {
                        vector<float> blockHist;

                        // 각 블록 내에서 cellSize만큼 이동
                        for (int ci = 0; ci < blockSize.height; ci += cellSize.height) {
                            for (int cj = 0; cj < blockSize.width; cj += cellSize.width) {
                                vector<float> cellHist(nbins, 0);


                                // 각 셀에 대한 히스토그램 계산
                                for (int y = i + bi + ci; y < i + bi + ci + cellSize.height && y < rows; ++y) {
                                    for (int x = j + bj + cj; x < j + bj + cj + cellSize.width && x < cols; ++x) {
                                        float mag = magOri.first[y * cols + x];
                                        float ang = magOri.second[y * cols + x];

                                        int bin = static_cast<int>(nbins * (ang >= 180 ? ang - 180 : ang) / 180.0) % nbins;
                                        cellHist[bin] += mag;
                                    }
                                }

                                /*
                               //Visualize Histogram

                                visualizeHistogram(cellHist);
                                waitKey(0);
                                */

                                // 셀 히스토그램 계산 완료하였으니 block 히스토그램에 추가
                                // 블록 히스토그램은 모든 셀들의 히스토그램이 저장됨
                                blockHist.insert(blockHist.end(), cellHist.begin(), cellHist.end());
                            }
                        }

                        // Normalize the block histogram
                        // L2 놈
                        float blockSum = 0.0;
                        for (float val : blockHist) {
                            blockSum += val * val;
                        }
                        blockSum = std::sqrt(blockSum) + 1e-6;
                        for (float& val : blockHist) {
                            val /= blockSum;
                        }

                        // 정규화된 한 블록을 windowHist에 추가
                        windowHist.insert(windowHist.end(), blockHist.begin(), blockHist.end());
                    }
                }
                // Add the window histogram to the final descriptor vector
                descriptors.insert(descriptors.end(), windowHist.begin(), windowHist.end());
            }
        }

        // Free dynamically allocated memory
        delete[] grayArray;
        delete[] edge.first;
        delete[] edge.second;
        delete[] magOri.first;
        delete[] magOri.second;
    }

private:
    Size winSize;
    Size blockSize;
    Size cellSize;
    Size winStride;
    Size blockStride;

    int nbins;
};

vector<float> HogFeatureExtractor(Mat colorFrame){
    // 방법2 (겹치게): 정확도 89.8925 %
    Size winSize(32, 32);
    Size blockSize(16, 16);
    Size cellSize(16, 16);
    Size winStride(16, 16);
    Size blockStride(8, 8);
    int nbins = 9;
    
    // Create HOG descriptor
    MyHOGDescriptor hog(winSize, blockSize, cellSize, winStride, blockStride,  nbins);
   
    // Compute HOG features
    vector<float> descriptors;
    hog.compute(colorFrame, descriptors);

    return descriptors;
}