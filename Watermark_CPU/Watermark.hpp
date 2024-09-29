#pragma once

#include "eigen_rgb_array.hpp"
#include <Eigen/Dense>
#include <string>

enum MASK_TYPE 
{
	ME,
	NVF
};

class Watermark {
private:
	Eigen::ArrayXXf randomMatrix;
	int p, pSquared, halfNeighborsSize, pad;
	Eigen::Index rows, cols, paddedRows, paddedCols;
	Eigen::ArrayXXf padded;
	float strengthFactor;

	void createNeighbors(const Eigen::ArrayXXf& array, Eigen::VectorXf& x_, const int neighborSize, const int i, const int j) const;
	Eigen::ArrayXXf loadRandomMatrix(const std::string wFilePath, const Eigen::Index rows, const Eigen::Index cols) const;
	Eigen::ArrayXXf computeCustomMask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded) const;
	Eigen::ArrayXXf computePredictionErrorMask(const Eigen::ArrayXXf& paddedImage, Eigen::ArrayXXf& errorSequence, Eigen::VectorXf& coefficients, const bool maskNeeded) const;
	Eigen::ArrayXXf computeErrorSequence(const Eigen::ArrayXXf& padded, const Eigen::VectorXf& coefficients) const;

public:
	Watermark(const Eigen::Index rows, const Eigen::Index cols, const std::string wFilePath, const int p, const float psnr);
	Watermark(const Watermark& other) = default;
	Watermark& operator=(const Watermark& other) = default;
	Watermark(Watermark&& other) noexcept = default;
	Watermark& operator=(Watermark&& other) noexcept = default;
	EigenArrayRGB makeWatermark(const Eigen::ArrayXXf& inputImage, const EigenArrayRGB& outputImage, MASK_TYPE type);
	float detectWatermark(const Eigen::ArrayXXf& watermarkedImage, MASK_TYPE type);
};