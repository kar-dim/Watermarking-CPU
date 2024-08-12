#include "Utilities.hpp"
#include <chrono>
#include <string>
#include "tensor_types.hpp"
#include "cimg_init.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <omp.h>
#include <utility>

using std::string;
using namespace cimg_library;
using namespace Eigen;

string add_suffix_before_extension(const string& file, const string& suffix) {
	auto dot = file.find_last_of('.');
	return dot == string::npos ? file + suffix : file.substr(0, dot) + suffix + file.substr(dot);
}

CImg<float> eigen_tensor_to_cimg(const Tensor3d& tensor_rgb, const float clamp_low, const float clamp_high) {
	const int rows = static_cast<int>(tensor_rgb.dimension(0));
	const int cols = static_cast<int>(tensor_rgb.dimension(1));
	const int channels = static_cast<int>(tensor_rgb.dimension(2));
	CImg<float> cimg_image(cols, rows, 1, channels);
#pragma omp parallel for
	for (int y = 0; y < rows; ++y)
		for (int x = 0; x < cols; ++x)
			for (int c = 0; c < channels; c++)
				cimg_image(x, y, 0, c) = std::max(clamp_low, std::min(tensor_rgb(y, x, c), clamp_high));
	return cimg_image;
}

Tensor3d cimg_to_eigen_tensor(CImg<float>& rgb_image) {
	Tensor3d tensor_rgb(rgb_image.height(), rgb_image.width(), rgb_image.spectrum());
#pragma omp parallel for
	for (int y = 0; y < rgb_image.height(); y++)
		for (int x = 0; x < rgb_image.width(); x++)
			for (int c = 0; c < rgb_image.spectrum(); c++)
				tensor_rgb(y, x, c) = rgb_image(x, y, 0, c);
	return tensor_rgb;
}

ArrayXXf eigen_tensor_to_grayscale_array(const Tensor3d& tensor_rgb, const float r_weight, const float g_weight, const float b_weight) {
	const auto rows = tensor_rgb.dimension(0);
	const auto cols = tensor_rgb.dimension(1);
	Tensor2d tensor_2d = tensor_rgb.chip(0, 2) * r_weight + tensor_rgb.chip(1, 2) * g_weight + tensor_rgb.chip(2, 2) * b_weight;
	return ArrayXXf::Map(tensor_2d.data(), rows, cols);
}

//χρονομέτρηση
namespace timer {
	void start() {
		start_timex = std::chrono::high_resolution_clock::now();
	}
	void end() {
		cur_timex = std::chrono::high_resolution_clock::now();
	}
	double secs_passed() {
		return (double)std::chrono::duration_cast<std::chrono::microseconds>(cur_timex - start_timex).count() / 1000000;
	}
}