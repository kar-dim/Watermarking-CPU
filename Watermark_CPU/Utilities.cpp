#include "cimg_init.hpp"
#include "eigen_rgb_array.hpp"
#include "Utilities.hpp"
#include <chrono>
#include <Eigen/Dense>
#include <string>

using namespace cimg_library;
using namespace Eigen;

using std::string;

string addSuffixBeforeExtension(const string& file, const string& suffix) 
{
	auto dot = file.find_last_of('.');
	return dot == string::npos ? file + suffix : file.substr(0, dot) + suffix + file.substr(dot);
}

CImg<float> eigen3dArrayToCimg(const EigenArrayRGB& arrayRgb) 
{
	const auto rows = arrayRgb[0].rows();
	const auto cols = arrayRgb[0].cols();
	CImg<float> cimg_image(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows), 1, 3);
	//a parallel pixel by pixel copy for loop is faster instead of three parallel (channel) bulk memory copies
	//because cimg and eigen use different memory layouts, and transposing is required which would make the copy much slower
#pragma omp parallel for
	for (int y = 0; y < rows; ++y)
		for (int x = 0; x < cols; ++x)
			for (int channel = 0; channel < 3; channel++)
				cimg_image(x, y, 0, channel) = arrayRgb[channel](y, x);
	return cimg_image;
}

EigenArrayRGB cimgToEigen3dArray(const CImg<float>& rgbImage) 
{
	const int rows = rgbImage.height();
	const int cols = rgbImage.width();
	//a parallel pixel by pixel copy for loop is faster instead of three parallel (channel) bulk memory copies
	//because cimg and eigen use different memory layouts, and transposing is required which would make the copy much slower
	EigenArrayRGB rgb_array = { ArrayXXf(rows,cols), ArrayXXf(rows,cols), ArrayXXf(rows, cols) };
#pragma omp parallel for
	for (int y = 0; y < rgbImage.height(); y++)
		for (int x = 0; x < rgbImage.width(); x++)
			for (int channel = 0; channel < 3; channel++)
				rgb_array[channel](y,x) = rgbImage(x, y, 0, channel);
	return rgb_array;
}

ArrayXXf eigen3dArrayToGrayscaleArray(const EigenArrayRGB& arrayRgb, const float rWeight, const float gWeight, const float bWeight) 
{
	return (arrayRgb[0] * rWeight) + (arrayRgb[1] * gWeight) + (arrayRgb[2] * bWeight);
}

//χρονομέτρηση
namespace timer 
{
	void start() 
	{
		startTime = std::chrono::high_resolution_clock::now();
	}
	void end() 
	{
		currentTime = std::chrono::high_resolution_clock::now();
	}
	double elapsedSeconds() 
	{
		return (double)std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime).count() / 1000000;
	}
}