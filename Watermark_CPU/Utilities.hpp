#pragma once
#include "eigen_rgb_array.hpp"
#include <chrono>
#include <CImg.h>
#include <Eigen/Dense>
#include <string>

std::string addSuffixBeforeExtension(const std::string& file, const std::string& suffix);
cimg_library::CImg<float> eigen3dArrayToCimg(const EigenArrayRGB& imageRgb);
EigenArrayRGB cimgToEigen3dArray(const cimg_library::CImg<float>& rgbImage);
Eigen::ArrayXXf eigen3dArrayToGrayscaleArray(const EigenArrayRGB& imageRgb, const float rWeight, const float gWeight, const float bWeight);

namespace timer 
{
	static std::chrono::time_point<std::chrono::steady_clock> startTime, currentTime;
	void start();
	void end();
	double elapsedSeconds();
}