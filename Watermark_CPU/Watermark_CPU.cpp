#pragma warning(disable:4996)
#define _CRT_SECURE_NO_WARNINGS
#include "INIReader.h"
#define cimg_use_cpp11 1
#define cimg_use_png
#include "CImg.h"
#include "UtilityFunctions.h"
#include "WatermarkFunctions.h"
#include <iostream>
#include <thread>
#include <omp.h>
#include <Eigen/Dense>
#include <iomanip>
#include <string>
#include <cmath>
#include <memory>
#include <string>
using namespace cimg_library;
using std::cout;

/*!
 *  \brief  This is a project implementation of my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU
 *  \author Dimitris Karatzas
 */
int main(int argc, char** argv)
{
	INIReader inir("settings.ini");
	if (inir.ParseError() < 0) {
		cout << "Could not load configuration file, exiting..";
		return -1;
	}
	const char *image_path = strdup(inir.Get("paths", "image", "NO_IMAGE").c_str());
	const int p = inir.GetInteger("parameters", "p", 5);
	const float psnr = static_cast<float>(inir.GetReal("parameters", "psnr", 30.0f));
	const std::string w_file = inir.Get("paths", "w_path", "w.txt");
	const int loops = inir.GetInteger("parameters", "loops_for_test", 5);
	int num_threads = inir.GetInteger("parameters", "threads", 0);
	if (num_threads <= 0 || num_threads > 256) {
		auto threads_supported = std::thread::hardware_concurrency();
		num_threads = threads_supported == 0 ? 2 : threads_supported;
	}
	cout << "Using " << num_threads << " parallel threads.\n";
	omp_set_num_threads(num_threads);
	//openmp initialization
#pragma omp parallel for
	for (int i = 0; i < 24; i++) {}

	cout << "Each test will be executed " << loops << " times. Average time will be shown below\n";

	CImg<float> rgb_image(image_path);
	const int rows = rgb_image.height();
	const int cols = rgb_image.width();
	const int elems = rows * cols;

	if (cols <= 16 || rows <= 16 || rows >= 16384 || cols >= 16384) {
		cout << "Image dimensions too low or too high\n";
		return -1;
	}
	if (p <= 0 || p % 2 != 1 || p > 9) {
		cout << "p parameter must be a positive odd number less than 9\n";
		return -1;
	}
	if (psnr <= 0) {
		cout << "PSNR must be a positive number\n";
		return -1;
	}
	cout << "Image size is: " << rows << " rows and " << cols << " columns\n\n";
	auto grayscale_vals = std::unique_ptr<float>(new float[elems]);
	for (int i = 0; i < elems; i++)
		grayscale_vals.get()[i] = static_cast<float>(std::round(0.299 * rgb_image.data()[i]) + std::round(0.587 * rgb_image.data()[i + elems]) + std::round(0.114 * rgb_image.data()[i + 2 * elems]));

	Eigen::ArrayXXf image_m = Eigen::Map<Eigen::ArrayXXf>(grayscale_vals.get(), cols, rows);
	image_m.transposeInPlace();

	//tests begin
	try {
		//initialize main class responsible for watermarking and detection
		WatermarkFunctions watermarkFunctions(image_m, w_file, p, psnr);

		double secs = 0;
		//NVF mask calculation
		Eigen::ArrayXXf image_m_nvf, image_m_me;
		for (int i = 0; i < loops; i++) {
			timer::start();
			image_m_nvf = watermarkFunctions.make_and_add_watermark_NVF();
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate NVF mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / loops << " seconds.\n\n";

		secs = 0;
		//Prediction error mask calculation
		for (int i = 0; i < loops; i++) {
			timer::start();
			image_m_me = watermarkFunctions.make_and_add_watermark_prediction_error();
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate ME mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / loops << " seconds.\n\n";

		float correlation_nvf, correlation_me;
		secs = 0;
		//NVF mask detection
		for (int i = 0; i < loops; i++) {
			timer::start();
			correlation_nvf = watermarkFunctions.mask_detector(image_m_nvf, MASK_TYPE::NVF);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate NVF [COR] of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / loops << " seconds.\n\n";

		secs = 0;
		//Prediction error mask detection
		for (int i = 0; i < loops; i++) {
			timer::start();
			correlation_me = watermarkFunctions.mask_detector(image_m_me, MASK_TYPE::ME);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate ME [COR] of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / loops << " seconds.\n\n";

		cout << "Correlation [NVF]: " << std::fixed << std::setprecision(16) << correlation_nvf << "\n";
		cout << "Correlation [ME]: " << std::fixed << std::setprecision(16) << correlation_me << "\n";
	}
	catch (const std::exception& e) {
		cout << e.what() << "\n";
	}

	return 0;
}