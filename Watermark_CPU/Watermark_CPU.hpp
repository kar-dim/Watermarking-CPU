#pragma once
#include "eigen_rgb_array.hpp"
#include <string>

std::string execution_time(const bool show_fps, const double seconds);
void exit_program(const int exit_code);
void save_watermarked_image(const std::string& image_path, const std::string& suffix, const EigenArrayRGB& watermark);