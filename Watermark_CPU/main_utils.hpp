#pragma once
#include "eigen_rgb_array.hpp"
#include <string>

enum IMAGE_TYPE
{
	JPG,
	PNG
};

std::string executionTime(const bool showFps, const double seconds);
void exitProgram(const int exitCode);
void saveWatermarkedImage(const std::string& imagePath, const std::string& suffix, const EigenArrayRGB& watermark, const IMAGE_TYPE type);