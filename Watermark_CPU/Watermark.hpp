#pragma once

#include "eigen_rgb_array.hpp"
#include <Eigen/Dense>
#include <string>

enum MASK_TYPE {
	ME,
	NVF
};

class Watermark {
private:
	const Eigen::ArrayXXf randomMatrix;
	const int p, pSquared, halfNeighborsSize, pad;
	const Eigen::Index rows, cols, paddedRows, paddedCols;
	const float strengthFactor;

	void createNeighbors(const Eigen::ArrayXXf& array, Eigen::VectorXf& x_, const int neighborSize, const int i, const int j) const;
	Eigen::ArrayXXf loadRandomMatrix(const std::string wFilePath, const Eigen::Index rows, const Eigen::Index cols) const;
	Eigen::ArrayXXf computeCustomMask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded) const;
	Eigen::ArrayXXf computePredictionErrorMask(const Eigen::ArrayXXf& paddedImage, Eigen::ArrayXXf& errorSequence, Eigen::VectorXf& coefficients, const bool maskNeeded) const;
	Eigen::ArrayXXf computeErrorSequence(const Eigen::ArrayXXf& padded, const Eigen::VectorXf& coefficients) const;

public:
	Watermark(const Watermark& other) = delete;
	Watermark& operator=(const Watermark& other) = delete;
	Watermark(Watermark&& other) noexcept = delete;
	Watermark& operator=(Watermark&& other) = delete;

	Watermark(const Eigen::Index rows, const Eigen::Index cols, const std::string wFilePath, const int p, const float psnr);
	EigenArrayRGB makeWatermark(const Eigen::ArrayXXf& inputImage, const EigenArrayRGB& outputImage, MASK_TYPE type) const;
	float detectWatermark(const Eigen::ArrayXXf& watermarkedImage, MASK_TYPE type) const;
};