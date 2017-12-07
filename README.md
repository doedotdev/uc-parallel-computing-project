# Uncanny - Parallel Computing Final Assignment
Benjamin Horn - hornbd@mail.uc.edu

Chris Klammer - klammecr@mail.uc.edu

## Final Project Report
[Final Project Report Link](https://docs.google.com/document/d/1kwieEjQPy-1_5hu6fqBapzr9CZg_Kduay8AGREue0_o/edit?usp=sharing)

## Code BreakDown

The main functionality is among these three files.

 - [Compare main.cpp](https://github.com/hornbd96/uncanny/blob/master/compare/main.cpp)
 - [Serial serial.cpp](https://github.com/hornbd96/uncanny/blob/master/compare/main.cpp)

## Background
Our motivation for choosing edge detection as a project topic was influenced by Chris’s focus of work at his future internship. He will likely be working with edge detection and image processing in general for images taken by probes in outer space. A project such as this serves as an excellent introduction to work that he will do in the future.

Additionally, we have been exposed to a background of CUDA from the Udacity courses; so, we will be using CUDA as our platform. Currently, there are many edge detection algorithms out there. Some of the edge detection algorithms we could run experiments on include:

 - Frie Chen
 - Canny
 - Robert's
 - Prewitt’s
 - Sobel Gradient

## Application Level Objectives
The goal of the project is to parallelize edge detection algorithms. Through this, we can compare the run times and efficiency of both the serial and parallel executions to show the potential performance gain. Additionally, we could parallelize multiple edge detection algorithms to compare their performance. This will be done using the CUDA tools we previously learned. The final solution will be simple to implement in image processing applications.

The benefits of our end product will be a faster and more efficient edge detection process when done in parallel. The benefits can be used in many edge detection applications such as data extraction from images, image processing, and segmentation.

Must Have Functionalities and Features
 - Serial implementation of the canny edge detection algorithm.
 - Parallel implementation of the canny edge detection algorithm.
 - Comparison of performance between serial and parallel implementation of canny algorithm.
 - Document the findings and supporting visuals.

Optional Functionalities and Features
 - Explore additional edge detection algorithms if time permits.
 - Comparison of performance between serial and parallel implementation of these additional algorithms.
 - Comparison of performance between the multiple edge detection algorithms.
 - Document the additional findings and supporting visuals.

## Design Overview

[For full assignment click here.](https://docs.google.com/document/d/10MgV2Bd2kwDwLw-Y-uwgxVlDKIK5PxdYGMFzr4ZmF3M/edit?usp=sharing)

## Performance Goals and Validation
Minimum performance goals for our edge detection implementation is to be considerably faster than the serial implementation of the algorithm. Expectations for running our edge detection algorithm in parallel on the GPU using the CUDA platform are to be five times faster than the serial implementation at minimum.

Potential Bottlenecks:
 - Keeping the entirety of the image in global memory
 - Includes the copying from shared memory to global memory
 - Edge pixel candidates must be stored in global memory for Non-Maximum Suppression
 - How we tile the image for each thread block to work on 

 To test our implementation and ensure it is robust and correct we will compare the resulting output to our serial implementation of the algorithm. Through this, we can ensure that the performance gain of the parallel execution maintains the same level of quality. 

## Schedule and Division of Work
Benjamin Horn - Responsibilities
 - Serial implementation of the canny edge detection algorithm
 - Project Documentation
 - Performance comparison between serial and parallel runs
 - Implement image comparison to check CPU vs GPU output images

Chris Klammer - Responsibilities
 - CUDA implementation for Canny Edge Detection algorithm
 - Review Project Documentation
 - Performance comparison between different edge detection algorithms (if needed)
