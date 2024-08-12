#pragma warning(disable:4996)
#define _CRT_SECURE_NO_WARNINGS
#include "Watermark_CPU.hpp"
#include "Watermark.hpp"
#include "Utilities.hpp"
#include "INIReader.h"
#include <Eigen/Dense>
#include "cimg_init.hpp"
#include <iostream>
#include <thread>
#include <omp.h>
#include <iomanip>
#include <string>
#include <cstdlib>

#define R_WEIGHT 0.299f
#define G_WEIGHT 0.587f
#define B_WEIGHT 0.114f

using namespace cimg_library;
using namespace Eigen;
using std::cout;
using std::string;

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
		exit_program(EXIT_FAILURE);
	}
	const string image_path = inir.Get("paths", "image", "NO_IMAGE");
	const int p = inir.GetInteger("parameters", "p", 5);
	const float psnr = static_cast<float>(inir.GetReal("parameters", "psnr", 30.0f));
	const string w_file = inir.Get("paths", "w_path", "w.txt");
	int num_threads = inir.GetInteger("parameters", "threads", 0);
	if (num_threads <= 0 || num_threads > 256) {
		auto threads_supported = std::thread::hardware_concurrency();
		num_threads = threads_supported == 0 ? 2 : threads_supported;
	}
	int loops = inir.GetInteger("parameters", "loops_for_test", 5); 
	loops = loops <= 0 || loops > 64 ? 5 : loops;

	//openmp initialization
	omp_set_num_threads(num_threads);
#pragma omp parallel for
	for (int i = 0; i < 24; i++) {}

	CImg<float> rgb_image(image_path.c_str());
	const int rows = rgb_image.height();
	const int cols = rgb_image.width();
	const int elems = rows * cols;

	if (cols <= 16 || rows <= 16 || rows >= 16384 || cols >= 16384) {
		cout << "Image dimensions too low or too high\n";
		exit_program(EXIT_FAILURE);
	}
	if (p <= 0 || p % 2 != 1 || p > 9) {
		cout << "p parameter must be a positive odd number less than 9\n";
		exit_program(EXIT_FAILURE);
	}
	if (psnr <= 0) {
		cout << "PSNR must be a positive number\n";
		exit_program(EXIT_FAILURE);
	}

	cout << "Using " << num_threads << " parallel threads.\n";
	cout << "Each test will be executed " << loops << " times. Average time will be shown below\n";
	cout << "Image size is: " << rows << " rows and " << cols << " columns\n\n";

	//copy from cimg to Eigen
	timer::start();
	const Tensor3d tensor_rgb = cimg_to_eigen_tensor(rgb_image);
	const ArrayXXf array_grayscale = eigen_tensor_to_grayscale_array(tensor_rgb, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	timer::end();
	cout << "Time to load image from disk and initialize Cimg and Eigen memory objects: " << timer::secs_passed() << " seconds\n\n";
	
	//tests begin
	try {
		//initialize main class responsible for watermarking and detection
		Watermark watermark_obj(tensor_rgb, array_grayscale, w_file, p, psnr);

		double secs = 0;
		//NVF mask calculation
		Tensor3d watermark_NVF, watermark_ME;
		for (int i = 0; i < loops; i++) {
			timer::start();
			watermark_NVF = watermark_obj.make_and_add_watermark(MASK_TYPE::NVF);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate NVF mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / loops << " seconds.\n\n";
		
		secs = 0;
		//Prediction error mask calculation
		for (int i = 0; i < loops; i++) {
			timer::start();
			watermark_ME = watermark_obj.make_and_add_watermark(MASK_TYPE::ME);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate ME mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / loops << " seconds.\n\n";

		const ArrayXXf watermarked_NVF_gray = eigen_tensor_to_grayscale_array(watermark_NVF, R_WEIGHT, G_WEIGHT, B_WEIGHT);
		const ArrayXXf watermarked_ME_gray = eigen_tensor_to_grayscale_array(watermark_ME, R_WEIGHT, G_WEIGHT, B_WEIGHT);

		float correlation_nvf, correlation_me;
		secs = 0;
		//NVF mask detection
		for (int i = 0; i < loops; i++) {
			timer::start();
			correlation_nvf = watermark_obj.mask_detector(watermarked_NVF_gray, MASK_TYPE::NVF);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate NVF [COR] of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / loops << " seconds.\n\n";

		secs = 0;
		//Prediction error mask detection
		for (int i = 0; i < loops; i++) {
			timer::start();
			correlation_me = watermark_obj.mask_detector(watermarked_ME_gray, MASK_TYPE::ME);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate ME [COR] of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / loops << " seconds.\n\n";

		cout << "Correlation [NVF]: " << std::fixed << std::setprecision(16) << correlation_nvf << "\n";
		cout << "Correlation [ME]: " << std::fixed << std::setprecision(16) << correlation_me << "\n";

		//save watermarked images to disk
		if (inir.GetBoolean("options", "save_watermarked_files_to_disk", false)) {
			cout << "\nSaving watermarked files to disk...\n";
#pragma omp parallel sections 
	{
#pragma omp section
		{
			string watermarked_file = add_suffix_before_extension(image_path, "_W_NVF");
			auto cimg_array_to_save = eigen_tensor_to_cimg(watermark_NVF, 0.0f, 255.0f);
			cimg_array_to_save.save_png(watermarked_file.c_str());
			}
#pragma omp section
			{
			string watermarked_file = add_suffix_before_extension(image_path, "_W_ME");
			auto cimg_array_to_save = eigen_tensor_to_cimg(watermark_ME, 0.0f, 255.0f);
			cimg_array_to_save.save_png(watermarked_file.c_str());
			}
	}
			cout << "Successully saved to disk\n";
		}
	}
	catch (const std::exception& e) {
		cout << e.what() << "\n";
		exit_program(EXIT_FAILURE);
	}
	exit_program(EXIT_SUCCESS);
}

void exit_program(const int exit_code) {
	std::system("pause");
	std::exit(exit_code);
}