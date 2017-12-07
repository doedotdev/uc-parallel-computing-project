#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv; // imread // imwrite
using namespace std; // cout + endl


int main(int argc, char **argv) {

    // open cv implementation
    std::clock_t start;
    double totalTime;
    start = std::clock();

    String image1 = "turk.jpg";

    // ./programName would be argv[0]
    if (argv[1]) {
        image1 = argv[1];
    }
    cv::Mat inputImage = imread(image1, CV_LOAD_IMAGE_COLOR);
    cv::Mat outputImage;

    // Convert image to Greyscale
    cv::cvtColor(inputImage, outputImage, CV_BGR2GRAY);

    // image smoothing with gaussian blur
    // the more smoothing done, the clearer the final image will be
    // for(int i = 0; i < 30; i++) {
        cv::GaussianBlur(outputImage, outputImage, cv::Size(7, 7), 1.5, 1.5);
    //}

    // edge deteciton algorithm
    cv::Canny(outputImage, outputImage, 0, 30, 3);

    // Output time
    totalTime = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout << totalTime << std::endl;

    imshow("Output Image", outputImage);
    imwrite("OutputImage.png", outputImage);

    // image will auto close
    waitKey(0); // do not close window until done viewing image

    // end open cv implementation

    return 0;
}