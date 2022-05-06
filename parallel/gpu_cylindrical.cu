__device__ __inline__ Point2f convert_pt(Point2f point,int w,int h)
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

__global__ void transform_pixel(
    cudev::PtrStepSz<uchar> img, 
    cudev::PtrStepSz<uchar> outimg
    int width, int height
) {
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelX < width && pixelY < height)
    {
        current_pos = convert_pt(pixelX, pixelY, width, height);
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

        // uchar valueR =   weight_tl * img.at<Vec3b>(top_left)[0] +
        // weight_tr * img.at<Vec3b>(top_left.y,top_left.x+1)[0] +
        // weight_bl * img.at<Vec3b>(top_left.y+1,top_left.x)[0] +
        // weight_br * img.at<Vec3b>(top_left.y+1,top_left.x+1)[0];

        // uchar valueG =   weight_tl * img.at<Vec3b>(top_left)[1] +
        // weight_tr * img.at<Vec3b>(top_left.y,top_left.x+1)[1] +
        // weight_bl * img.at<Vec3b>(top_left.y+1,top_left.x)[1] +
        // weight_br * img.at<Vec3b>(top_left.y+1,top_left.x+1)[1];

        // uchar valueB =   weight_tl * img.at<Vec3b>(top_left)[2] +
        // weight_tr * img.at<Vec3b>(top_left.y,top_left.x+1)[2] +
        // weight_bl * img.at<Vec3b>(top_left.y+1,top_left.x)[2] +
        // weight_br * img.at<Vec3b>(top_left.y+1,top_left.x+1)[2];

        uchar valueR = weight_tl * img.ptr()

        // tmpimg.at<Vec3b>(y,x)[0] = valueR;
        // tmpimg.at<Vec3b>(y,x)[1] = valueG;
        // tmpimg.at<Vec3b>(y,x)[2] = valueB;
        outimg.ptr(pixelY)[pixelX][0] = valueR;
        outimg.ptr(pixelY)[pixelX][1] = valueG;
        outimg.ptr(pixelY)[pixelX][2] = valueB;
    }
}

Mat cylindrical(Mat& img){

    //convert to cuda images
    cuda::GpuMat tmpimg(img);
    cuda::GpuMat outimg(img.size(), img.type());

    int width=img.cols;
    int height=img.rows;
    int N = width * height;
    
    double xCil, yCil, xImg, yImg;
    Mat result(img.size(),CV_8UC3);

    // int minx=img.cols /2;
    // int maxx=img.cols/2;
    // int miny=img.rows/2;
    // int maxy=img.rows/2;

    dim3 blockDim(BLOCKWIDTH, BLOCKWIDTH);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y -1) / blockDim.y
    );
    transform_pixel<<<gridDim, blockDim>>>(tmpimg, outimg, width, height);

    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());

    outimg.download(result)
	return img;
}


void main() {
    #pragma omp parallel for default(shared)
    for (int i = 0; i < numOfImages; i++) {
        Mat img = imgs_color[i];        
       
        resize(img, img, Size(img.cols / 3, img.rows / 3));
        copyMakeBorder(img, img, 100, 100, 100, 100, BORDER_CONSTANT);
        
        

        cylindrical(tmpimg, outimg);

        

        imgs_color[i] = img;
        cvtColor(img, img, COLOR_BGR2GRAY);
        imgs[i] = img;
    }
}