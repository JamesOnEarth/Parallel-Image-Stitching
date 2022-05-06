int main()
{
    omp_set_num_threads(THREAD_NUM);

    int numOfImages = 12;
    // const char *images[numOfImages] = {"images/1.jpg", "images/2.jpg", "images/3.jpg", "images/4.jpg"};
    const char *images[numOfImages] = {"uc/uc_1.jpg", "uc/uc_2.jpg", "uc/uc_3.jpg", "uc/uc_4.jpg", "uc/uc_5.jpg", "uc/uc_6.jpg", "uc/uc_7.jpg", "uc/uc_8.jpg", "uc/uc_9.jpg", "uc/uc_10.jpg", "uc/uc_11.jpg", "uc/uc_12.jpg"};
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
    
    // for (int i = numOfImages; i > 1; i /= 2){
        for (int i = 0; i < numOfImages / 2; i++){
            homographies[i*2].ptr<double>(0)[2] += dx;
            homographies[i*2].ptr<double>(1)[2] += dy;
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
    // }


    auto stitchStop = high_resolution_clock::now();

    stitchDuration = duration_cast<milliseconds>(stitchStop - stitchStart).count();

    auto cropStart = high_resolution_clock::now();

    result = cropImage(result);

    auto cropEnd = high_resolution_clock::now();

    cropDuration = duration_cast<milliseconds>(cropEnd - cropStart).count();

    auto compEnd = high_resolution_clock::now();

    // imwrite("images/parallel.jpg", result);
    imwrite("uc/parallel.jpg", result);

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