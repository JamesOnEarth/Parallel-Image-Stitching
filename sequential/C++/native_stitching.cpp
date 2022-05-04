#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/stitching.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    vector<Mat> imgs;
    int numOfImages = 4;
    const char *images[numOfImages] = {"images/1.jpg", "images/2.jpg", "images/3.jpg", "images/4.jpg"};

    for (int i = 0; i < numOfImages; i++) {
        Mat img = imread(images[i]);
        imgs.push_back(img);
    }

    Mat pano;
    Stitcher::Mode mode = Stitcher::PANORAMA;
    Ptr<Stitcher> stitcher = Stitcher::create(mode);
    Stitcher::Status status = stitcher->stitch(imgs, pano);
    imwrite("images/native_result.jpg", pano);
}