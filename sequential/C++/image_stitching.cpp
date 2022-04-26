#include<iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp> 

using namespace std;
using namespace cv;

void cropImage(Mat &img)
{
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat thresh;
    threshold(gray, thresh, 0, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    img = img(boundingRect(contours[0]));
}

void findKeyPoints(Mat img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
    Ptr<SIFT> sift = SIFT::create();
    sift->detectAndCompute(img, noArray(), keypoints, descriptors);
}

void matchKeyPoints(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &matches)
{
    Ptr<BFMatcher> matcher = BFMatcher::create();
    vector<vector<DMatch>> tmp_matches;
    //query descriptor, train descriptor
    matcher->knnMatch(descriptors1, descriptors2, tmp_matches, 2);

    for (int i = 0; i < tmp_matches.size(); i++)
    {
        if (tmp_matches[i][0].distance < 0.75f * tmp_matches[i][1].distance)
        {
            matches.push_back(tmp_matches[i][0]);
        }
    }
}

void getHomography(vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<DMatch>& matches, Mat &homography)
{
    if (matches.size() < 4)
    {
        cout << "Not enough matches" << endl;
        return;
    }
    vector<Point2f> points1, points2;
    for (int i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    homography = findHomography(points2, points1, RANSAC, 4.0);
}

int main()
{
    Mat image1_color = imread("images/1.jpg");
    Mat image2_color = imread("images/2.jpg");

    Mat image1, image2;
    cvtColor(image1_color, image1, COLOR_BGR2GRAY);
    cvtColor(image2_color, image2, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    findKeyPoints(image1, keypoints1, descriptors1);
    findKeyPoints(image2, keypoints2, descriptors2);
    

    vector<DMatch> matches;
    matchKeyPoints(descriptors1, descriptors2, matches);

    Mat homography;
    getHomography(keypoints1, keypoints2, matches, homography);
    int width = image1_color.cols + image2_color.cols;
    int height = image1_color.rows + image2_color.rows;
    Mat result = Mat::zeros(height, width, CV_8UC3);
    warpPerspective(image2_color, result, homography, result.size());
    image1_color.copyTo(result(Rect(0, 0, image1_color.cols, image1_color.rows)));

    cropImage(result);
    imwrite("images/result.jpg", result);
}