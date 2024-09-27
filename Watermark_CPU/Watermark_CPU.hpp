#pragma once
#include "eigen_rgb_array.hpp"
#include <string>

std::string execution_time(const bool showFps, const double seconds);
void exitProgram(const int exitCode);
void saveWatermarkedImage(const std::string& imagePath, const std::string& suffix, const EigenArrayRGB& watermark);