#pragma once
#include <string>
#include "cimg_init.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>

using std::string;
using namespace cimg_library;
using namespace Eigen;

typedef Eigen::Tensor<float, 2> Tensor2d;
typedef Eigen::Tensor<float, 3> Tensor3d;

string add_suffix_before_extension(const string& file, const string& suffix);
CImg<float> eigen_tensor_to_cimg(const Tensor3d& image_rgb, const float clamp_low, const float clamp_high);
Tensor3d cimg_to_eigen_tensor(CImg<float>& rgb_image);
ArrayXXf eigen_tensor_to_grayscale_array(const Tensor3d& image_rgb, const float r_weight, const float g_weight, const float b_weight);

namespace timer {
	static std::chrono::time_point<std::chrono::steady_clock> start_timex, cur_timex;
	void start();
	void end();
	double secs_passed();
}