#include <iostream>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp> 

using namespace std;
using namespace cv;
using namespace std::chrono;

#define THREAD_NUM 1
#define MAX_IMG 50

Point2f convert_pt(Point2f point,int w,int h)
{
    //center the point at 0,0
    Point2f pc(point.x-w/2,point.y-h/2);

    //these are your free parameters
    float f = -w/2;
    float r = w;

    float omega = w/2;
    float z0 = f - sqrt(r*r-omega*omega);

    float zc = (2*z0+sqrt(4*z0*z0-4*(pc.x*pc.x/(f*f)+1)*(z0*z0-r*r)))/(2* (pc.x*pc.x/(f*f)+1)); 
    Point2f final_point(pc.x*zc/f,pc.y*zc/f);
    final_point.x += w/2;
    final_point.y += h/2;
    return final_point;
}


Mat cylindrical(Mat& img){
    int width=img.cols;
    int height=img.rows;
    Mat tmpimg(img.size(),CV_8UC3);


    int minx=img.cols /2;
    int maxx=img.cols/2;
    int miny=img.rows/2;
    int maxy=img.rows/2;


    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            Point2f current_pos(x,y);
            current_pos = convert_pt(current_pos, width, height);

            Point2i top_left((int)current_pos.x,(int)current_pos.y); //top left because of integer rounding

            //make sure the point is actually inside the original image
            if(top_left.x < 0 ||
            top_left.x > width-2 ||
            top_left.y < 0 ||
            top_left.y > height-2)
            {
                continue;
            }

            //bilinear interpolation
            float dx = current_pos.x-top_left.x;
            float dy = current_pos.y-top_left.y;

            float weight_tl = (1.0 - dx) * (1.0 - dy);
            float weight_tr = (dx)       * (1.0 - dy);
            float weight_bl = (1.0 - dx) * (dy);
            float weight_br = (dx)       * (dy);

            uchar valueR =   weight_tl * img.at<Vec3b>(top_left)[0] +
            weight_tr * img.at<Vec3b>(top_left.y,top_left.x+1)[0] +
            weight_bl * img.at<Vec3b>(top_left.y+1,top_left.x)[0] +
            weight_br * img.at<Vec3b>(top_left.y+1,top_left.x+1)[0];

            uchar valueG =   weight_tl * img.at<Vec3b>(top_left)[1] +
            weight_tr * img.at<Vec3b>(top_left.y,top_left.x+1)[1] +
            weight_bl * img.at<Vec3b>(top_left.y+1,top_left.x)[1] +
            weight_br * img.at<Vec3b>(top_left.y+1,top_left.x+1)[1];

            uchar valueB =   weight_tl * img.at<Vec3b>(top_left)[2] +
            weight_tr * img.at<Vec3b>(top_left.y,top_left.x+1)[2] +
            weight_bl * img.at<Vec3b>(top_left.y+1,top_left.x)[2] +
            weight_br * img.at<Vec3b>(top_left.y+1,top_left.x+1)[2];

            if(valueR>0 || valueG > 0 || valueB > 0){
                maxx=max(maxx,x);
                minx=min(minx,x);
                maxy=max(maxy,y);
                miny=min(miny,y);
            }

            tmpimg.at<Vec3b>(y,x)[0] = valueR;
            tmpimg.at<Vec3b>(y,x)[1] = valueG;
            tmpimg.at<Vec3b>(y,x)[2] = valueB;
        }
    }

    tmpimg = tmpimg(Range(miny, maxy), Range(minx, maxx));

	return tmpimg;
}


Mat cropImage(Mat &img)
{
    int top=0;
    int bottom=img.rows;

	for(int i = 0; i < img.cols; i++){
		for(int j = 0; j < img.rows; j++){
			if(img.ptr<Vec3b>(j)[i][0] != 0 && img.ptr<Vec3b>(j)[i][1] != 0 && img.ptr<Vec3b>(j)[i][2] != 0){
				top = max(top,j);
				break;
			}
		}
	}

	for(int i = 0; i < img.cols; i++){
		for(int j = img.rows - 1; j >= 0; j--){
			if(img.ptr<Vec3b>(j)[i][0] != 0 && img.ptr<Vec3b>(j)[i][1] != 0 && img.ptr<Vec3b>(j)[i][2] != 0){
				bottom = min(bottom,j);
				break;
			}
		}
	}

	return img(Range(top,bottom),Range(0,img.cols));

}

void findKeyPoints(Mat img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
    Ptr<SIFT> sift = SIFT::create();
    sift->detectAndCompute(img, noArray(), keypoints, descriptors);
}

void matchKeyPoints(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &matches)
{
    FlannBasedMatcher matcher;
    vector<DMatch> tmp_matches;
    //query descriptor, train descriptor
    matcher.match(descriptors1, descriptors2, tmp_matches);

    double minDist = 100;
    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = tmp_matches[i].distance;
        if(dist < minDist) {
            minDist = dist;
        }
    }


    for (int i = 0; i < descriptors1.rows; i++)
    {
        if (tmp_matches[i].distance < max(5 * minDist, 0.02))
        {
            matches.push_back(tmp_matches[i]);
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

	/*
		Htmp: 
		[1 , 0 , tx]
		[0 , 1 , ty]
	*/
	Mat Htmp(2,3,CV_64F,Scalar(0));
	Htmp.ptr<double>(0)[0]=1;
	Htmp.ptr<double>(1)[1]=1;

	int matchnum=0;

	for(int i = 0;i < points2.size(); i++){
		double tx,ty;
		tx=points1[i].x-points2[i].x;
		ty=points1[i].y-points2[i].y;
		int sum=0;
		/*
			1.calculate this match point translation matrix
			2.use this matrix to translate other point
			3.calculate the error
		*/
		for(int j = 0;j < points2.size(); j++){   
			double difx=double(points2[j].x)+tx-double(points1[j].x);
			double dify=double(points2[j].y)+ty-double(points1[j].y);
			if(difx<0){
				difx*=(-1);
			}
			if(dify<0){
				dify*=(-1);
			}
			if(difx+dify<3){ // error < threshold 
				sum++;
			}
		}
		// sum bigger mean this translation matrix is better 
		if(sum>matchnum){
			matchnum=sum;
			Htmp.ptr<double>(0)[2]=tx;
			Htmp.ptr<double>(1)[2]=ty;
		}

	}
	homography = Htmp;
}


void stitch(Mat &src,Mat &warp,int midline){
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			if(j <= midline){
                warp.ptr<Vec3b>(i)[j][0]=src.ptr<Vec3b>(i)[j][0];
                warp.ptr<Vec3b>(i)[j][1]=src.ptr<Vec3b>(i)[j][1];
                warp.ptr<Vec3b>(i)[j][2]=src.ptr<Vec3b>(i)[j][2];
            }
        }
	}
}

int main()
{
    omp_set_num_threads(THREAD_NUM);

    int numOfImages = 16;
    // const char *images[numOfImages] = {"tepper/1.jpg", "tepper/2.jpg", "tepper/3.jpg", "tepper/4.jpg"};
    /* const char *images[numOfImages] = {"uc/uc_1.jpg", "uc/uc_2.jpg", "uc/uc_3.jpg", "uc/uc_4.jpg", 
                                          "uc/uc_5.jpg", "uc/uc_6.jpg", "uc/uc_7.jpg", "uc/uc_8.jpg",
                                          "uc/uc_9.jpg", "uc/uc_10.jpg", "uc/uc_11.jpg", "uc/uc_12.jpg"}; */
    const char *images[numOfImages] = {"lounge/lounge_1.jpg", "lounge/lounge_2.jpg", "lounge/lounge_3.jpg",
                                       "lounge/lounge_4.jpg", "lounge/lounge_5.jpg", "lounge/lounge_6.jpg",
                                       "lounge/lounge_7.jpg", "lounge/lounge_8.jpg", "lounge/lounge_9.jpg", 
                                       "lounge/lounge_10.jpg", "lounge/lounge_11.jpg", "lounge/lounge_12.jpg", 
                                       "lounge/lounge_13.jpg", "lounge/lounge_14.jpg", "lounge/lounge_15.jpg", 
                                       "lounge/lounge_16.jpg"}; 
    Mat imgs[MAX_IMG];
    Mat imgs_color[MAX_IMG];
    vector<KeyPoint> keypoints[MAX_IMG];
    Mat descriptors[MAX_IMG];

    int cylindricalDuration = 0;
    int keypointsDuration = 0;
    int matchDuration = 0;
    int transformDuration = 0; 
    int totalDuration = 0;
    int stitchDuration = 0;
    int cropDuration = 0;

    auto allStart = high_resolution_clock::now();

    for (int i = 0; i < numOfImages; i++) {
        imgs_color[i] = imread(images[i]);
    }

    auto compStart = high_resolution_clock::now();

    auto cylindricalStart = high_resolution_clock::now();

    #pragma omp parallel for default(shared)
    for (int i = 0; i < numOfImages; i++) {
        Mat img = imgs_color[i];        
       
        resize(img, img, Size(img.cols / 3, img.rows / 3));
        copyMakeBorder(img, img, 100, 100, 100, 100, BORDER_CONSTANT);
        
        img = cylindrical(img);
        imgs_color[i] = img;
        cvtColor(img, img, COLOR_BGR2GRAY);
        imgs[i] = img;
    }

    auto cylindricalStop  = high_resolution_clock::now();

    cylindricalDuration = duration_cast<milliseconds>(cylindricalStop - cylindricalStart).count();

    auto keypointStart = high_resolution_clock::now();

    #pragma omp parallel for default(shared) schedule(dynamic)
    for (int i = 0; i < numOfImages; i++) {
        findKeyPoints(imgs[i], keypoints[i], descriptors[i]);
    }

    auto keypointStop = high_resolution_clock::now();

    keypointsDuration = duration_cast<milliseconds>(keypointStop - keypointStart).count();

    vector<DMatch> matches[MAX_IMG - 1];
    auto matchStart = high_resolution_clock::now();

    #pragma omp parallel for default(shared) schedule(dynamic)
    for (int i = 0; i < numOfImages - 1; i++) {
        matchKeyPoints(descriptors[i], descriptors[i+1], matches[i]);
    }
    auto matchStop = high_resolution_clock::now();

    matchDuration = duration_cast<milliseconds>(matchStop - matchStart).count();

    Mat homographies[MAX_IMG - 1];

    auto transformStart = high_resolution_clock::now();

    #pragma omp parallel for default(shared) schedule(dynamic)
    for (int i = 0; i < numOfImages - 1; i++){
        getHomography(keypoints[i], keypoints[i+1], matches[i], homographies[i]);
    }

    auto transformStop = high_resolution_clock::now();

    transformDuration = duration_cast<milliseconds>(transformStop - transformStart).count();

    Mat result = imgs_color[0];
    int dx = 0;
    int dy = 0;

    auto stitchStart = high_resolution_clock::now();
    
    for (int i = 0; i < numOfImages - 1; i++){
        homographies[i].ptr<double>(0)[2] += dx;
		homographies[i].ptr<double>(1)[2] += dy;
		dx = homographies[i].ptr<double>(0)[2];
		dy = homographies[i].ptr<double>(1)[2];

        int mRows = max(result.rows, imgs_color[i+1].rows + int(homographies[i].ptr<double>(1)[2]));
		int mCols = imgs_color[i+1].cols + int(homographies[i].ptr<double>(0)[2]);
		int midline = (result.cols + int(homographies[i].ptr<double>(0)[2])) / 2;

        Mat warp = Mat::zeros(mRows, mCols, CV_8UC3);
        warpAffine(imgs_color[i+1], warp, homographies[i], Size(mCols, mRows));
		stitch(result, warp, midline);

        result = warp;    
    }

    auto stitchStop = high_resolution_clock::now();

    stitchDuration = duration_cast<milliseconds>(stitchStop - stitchStart).count();

    auto cropStart = high_resolution_clock::now();

    result = cropImage(result);

    auto cropEnd = high_resolution_clock::now();

    cropDuration = duration_cast<milliseconds>(cropEnd - cropStart).count();

    auto compEnd = high_resolution_clock::now();

    // imwrite("uc/parallel.jpg", result);
    // imwrite("tepper/parallel.jpg", result);
    imwrite("lounge/parallel.jpg", result);

    auto allEnd = high_resolution_clock::now();

    auto compDuration = duration_cast<milliseconds>(compEnd - compStart).count();
    auto duration = duration_cast<milliseconds>(allEnd - allStart).count();

    cout << "Total Elapsed Time: " << duration << " ms" << endl;
    cout << "Computational Time " << compDuration << " ms" << endl;
    cout << "Cylindrical: " << cylindricalDuration << " ms" << endl;
    cout << "Keypoints: " << keypointsDuration << " ms" << endl;
    cout << "Matching: " << matchDuration << " ms" << endl;
    cout << "Transform: " << transformDuration << " ms" << endl;
    cout << "Stitching: " << stitchDuration << " ms" << endl;
    cout << "Cropping " << cropDuration << " ms" << endl;
}