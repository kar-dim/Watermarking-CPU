#define cimg_use_png
#include "eigen_rgb_array.hpp"
#include "main_utils.hpp"
#include "Utilities.hpp"
#include "Watermark.hpp"
#include <CImg.h>
#include <cstdlib>
#include <Eigen/Dense>
#include <exception>
#include <INIReader.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <thread>

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
	const INIReader inir("settings.ini");
	if (inir.ParseError() < 0) 
	{
		cout << "Could not load configuration file, exiting..";
		exitProgram(EXIT_FAILURE);
	}
	const string imagePath = inir.Get("paths", "image", "NO_IMAGE");
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	const int p = inir.GetInteger("parameters", "p", 5);
	const float psnr = inir.GetFloat("parameters", "psnr", 30.0f);
	const string wFile = inir.Get("paths", "w_path", "w.txt");
	int numThreads = inir.GetInteger("parameters", "threads", 0);
	if (numThreads <= 0)
	{
		auto threadsSupported = std::thread::hardware_concurrency();
		numThreads = threadsSupported == 0 ? 2 : threadsSupported;
	}
	int loops = inir.GetInteger("parameters", "loops_for_test", 5); 
	loops = loops <= 0 || loops > 64 ? 5 : loops;

	//openmp initialization
	omp_set_num_threads(numThreads);
#pragma omp parallel for
	for (int i = 0; i < 24; i++) {}

	const CImg<float> rgbImageCimg(imagePath.c_str());
	const int rows = rgbImageCimg.height();
	const int cols = rgbImageCimg.width();

	if (cols <= 16 || rows <= 16 || rows >= 16384 || cols >= 16384) 
	{
		cout << "Image dimensions too low or too high\n";
		exitProgram(EXIT_FAILURE);
	}
	if (p <= 1 || p % 2 != 1 || p > 9) 
	{
		cout << "p parameter must be a positive odd number greater than or equal to 3 and less than or equal to 9\n";
		exitProgram(EXIT_FAILURE);
	}
	if (psnr <= 0) 
	{
		cout << "PSNR must be a positive number\n";
		exitProgram(EXIT_FAILURE);
	}

	cout << "Using " << numThreads << " parallel threads.\n";
	cout << "Each test will be executed " << loops << " times. Average time will be shown below\n";
	cout << "Image size is: " << rows << " rows and " << cols << " columns\n\n";

	//copy from cimg to Eigen
	timer::start();
	const EigenArrayRGB arrayRgb = cimgToEigen3dArray(rgbImageCimg);
	const ArrayXXf arrayGrayscale = eigen3dArrayToGrayscaleArray(arrayRgb, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	timer::end();
	cout << "Time to load image from disk and initialize CImg and Eigen memory objects: " << timer::elapsedSeconds() << " seconds\n\n";
	
	//tests begin
	try {
		//initialize main class responsible for watermarking and detection
		Watermark watermarkObj(rows, cols, wFile, p, psnr);
		float watermarkStrength;

		double secs = 0;
		//NVF mask calculation
		EigenArrayRGB watermarkNVF, watermarkME;
		for (int i = 0; i < loops; i++) 
		{
			timer::start();
			watermarkNVF = watermarkObj.makeWatermark(arrayGrayscale, arrayRgb, watermarkStrength, MASK_TYPE::NVF);
			timer::end();
			secs += timer::elapsedSeconds();
		}
		cout << "Watermark strength (parameter a): " << watermarkStrength <<"\nCalculation of NVF mask with " << rows << " rows and " << cols << " columns and parameters:\np = " << p << "  PSNR(dB) = " << psnr << "\n" << executionTime(showFps, secs / loops) << "\n\n";
		
		secs = 0;
		//Prediction error mask calculation
		for (int i = 0; i < loops; i++) 
		{
			timer::start();
			watermarkME = watermarkObj.makeWatermark(arrayGrayscale, arrayRgb, watermarkStrength, MASK_TYPE::ME);
			timer::end();
			secs += timer::elapsedSeconds();
		}
		cout << "Watermark strength (parameter a): " << watermarkStrength << "\nCalculation of ME mask with " << rows << " rows and " << cols << " columns and parameters:\np = " << p << "  PSNR(dB) = " << psnr << "\n" << executionTime(showFps, secs / loops) << "\n\n";

		const ArrayXXf watermarkedNVFgray = eigen3dArrayToGrayscaleArray(watermarkNVF, R_WEIGHT, G_WEIGHT, B_WEIGHT);
		const ArrayXXf watermarkedMEgray = eigen3dArrayToGrayscaleArray(watermarkME, R_WEIGHT, G_WEIGHT, B_WEIGHT);

		float correlationNvf, correlationMe;
		secs = 0;
		//NVF mask detection
		for (int i = 0; i < loops; i++) 
		{
			timer::start();
			correlationNvf = watermarkObj.detectWatermark(watermarkedNVFgray, MASK_TYPE::NVF);
			timer::end();
			secs += timer::elapsedSeconds();
		}
		cout << "Calculation of the watermark correlation (NVF) of an image with " << rows << " rows and " << cols << " columns and parameters:\np = " << p << "  PSNR(dB) = " << psnr << "\n" << executionTime(showFps, secs / loops) << "\n\n";

		secs = 0;
		//Prediction error mask detection
		for (int i = 0; i < loops; i++) 
		{
			timer::start();
			correlationMe = watermarkObj.detectWatermark(watermarkedMEgray, MASK_TYPE::ME);
			timer::end();
			secs += timer::elapsedSeconds();
		}
		cout << "Calculation of the watermark correlation (ME) of an image with " << rows << " rows and " << cols << " columns and parameters:\np = " << p << "  PSNR(dB) = " << psnr << "\n" << executionTime(showFps, secs / loops) << "\n\n";

		cout << "Correlation [NVF]: " << std::fixed << std::setprecision(16) << correlationNvf << "\n";
		cout << "Correlation [ME]: " << std::fixed << std::setprecision(16) << correlationMe << "\n";

		//save watermarked images to disk
		if (inir.GetBoolean("options", "save_watermarked_files_to_disk", false)) 
		{
			cout << "\nSaving watermarked files to disk...\n";
#pragma omp parallel sections 
			{
#pragma omp section
				saveWatermarkedImage(imagePath, "_W_NVF", watermarkNVF);
#pragma omp section
				saveWatermarkedImage(imagePath, "_W_ME", watermarkME);
			}
			cout << "Successully saved to disk\n";
		}
	}
	catch (const std::exception& e) {
		cout << e.what() << "\n";
		exitProgram(EXIT_FAILURE);
	}
	exitProgram(EXIT_SUCCESS);
}

//calculate execution time in seconds, or show FPS value
string executionTime(const bool showFps, const double seconds) 
{
	std::stringstream ss;
	if (showFps)
		ss << "FPS: " << std::fixed << std::setprecision(2) << 1.0 / seconds << " FPS";
	else
		ss << std::fixed << std::setprecision(6) << seconds << " seconds";
	return ss.str();
}

//save the provided Eigen RGB array containing a watermarked image to disk
void saveWatermarkedImage(const string& imagePath, const string& suffix, const EigenArrayRGB& watermark) 
{
	const string watermarkedFile = addSuffixBeforeExtension(imagePath, suffix);
	eigen3dArrayToCimg(watermark).save_png(watermarkedFile.c_str());
}

//exits the program with the provided exit code
void exitProgram(const int exitCode) 
{
	std::system("pause");
	std::exit(exitCode);
}