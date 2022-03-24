# Parallelizing Image Stitching

## Summary
This project will explore the level of parallelism that can be applied to various parts of the image stitching. We plan on implementing the image stitching workflow in C++ and CUDA.

## Background
To stitch two images together, we need to roughly follow the following workflow:
1. Find important keypoints and descriptors in images
2. Match these keypoints and descriptors across images
3. Align the images with homography transforms


The image stitching workflow begins by identifying key features in the images. Key points need to be detected in the images and this is commonly done using feature extractors such as SIFT, SUFT, and ORB. We will explore and compare feature extractors to determine which one can benefit the most from parallelization.

After identifying keypoints and key features in each image, we will need to match these points and features between different images. To do so, we need to make pairs between keypoints of images so that the Euclidian distance between keypoints are minimized. We will explore and compare FLANN matching and Brute Force matching for this subtask and explore which algorithm can benefit more from parallelism.

Having finally identified matching keypoints, the image stitching workflow then takes these matching points to calculate the homographic matrix that needs to be applied to the images such that they all align properly. The homographic matrix encompasses transformations such as rotation, translation, and perspective transformation. We will preliminarily explore the RANSAC (Random Sample Consensus) algorithm and investigate if it can benefit from parallelism. After calculation of the homographic matrix, the matrix is applied to the image to align the images. Images are then all stitched together to create the final panorama.


## The Challenge
As there are many stages of computation in the image stitching algorithm and sub-algorithms, there is bound to be a bottleneck or element that is hard to parallelize in one of the steps.

Keypoint Detection: 
There are two possible dimensions to parallel this subtask; we can parallelize between images, parallelize between partitions of an image, or a combination of both. Parallelizing between partitions of an image could have communication problems because there is dependency between different partitions of an image. We may also run into divergent execution with keypoint detection. Different images will likely have different number of keypoints, resulting in different execution times. Since three steps must be performed sequentially, we will have to synchronize at the end of detection, losing a bit of parallelism speedup.

Keypoint Matching:
There are again two possible dimensions to parallel this subtask; we can parallelize matching between image pairs, or parallelize between keypoint pairs across all images, or a combination of both. If parallelizing between image pairs, image pairs are bound to have less matching pairs of keypoints, so we would encounter divergent execution.

Homographic Transform:
The time required to calculate a homography matrix for a large number of points is very large. RANSAC is a randomized algorithm that samples points and continuously iterates to improve the homography matrix. This is going to be one of the major challenges of our project, as calculating the homography matrix directly for all points has a lot of dependencies and RANSAC is a naturally iterative converging algorithm.

Since we are working with pixels, we thankfully benefit from spatial locality. Because the three steps must be performed sequentially, we will require synchronization between the steps, which could lead to loss in speedup.

## Resources
We will be using OpenCV to implement an intitial sequential version just to get a rough idea of the entire workflow. We will also likely use OpenCV to generate tests cases which we will be using to evaluate the correctness of our result.

We will likely start on our parallel code from scratch, since it seems that all parts of the image stitching workflow could benefit from parallelizing. We will likely reference sequential implementations of the afromentioned algorithms. We do not currently believe we need any additional resources we currently do not have access to. Access to the Gates machine and our personal computers will likely suffice for the purpose of our project.



## Goals and Deliverables
We plan on completing a parallelized image stitching workflow using C++ and CUDA. We would like to compare the speedup and performance of our parallel version versus a sequential image stitching program. We will also compare the results of our parallel version to the sequential version to ensure the correctness of our result.

We plan on showing our results using input images and generated panorama that we produce. We can  create a image slideshow showing the before and after images.

If things go really well, we would like to have our stretch goal extend into video stitching. Video stitching is very similar to image stitching. We'd really only have to apply our current image stitching workflow across all the frames of the videos, so long as the videos are synced. It could grant us another dimension to work with, which would be interesting to parallelize.


## Platform Choice
We will be implementing our image stitching workflow using C++ and CUDA, because we think our workload should be convergent enough that it should benefit the most from CUDA programming model. 

We will be using the GHC machines and our local computers for our project because they both have access to a GPU which we can use to run our CUDA code. The GHC machines give us access to a RTX 2080 and our local computer has a RTX 2070 we can use to run our CUDA code.

## Schedule
week 1-2
 - Implement sequential version of code
 - Time the sections to determine what percentage of time is allocated to which section
 - Implement a baseline for parallel speedup comparison

week 3
 - Devise plans to find parallel solutions for each section

week 4
 - Implement parallel version of each section

week 5
 - Debugging parallel versions

week 6
 - Project paper, presentation, and poster
