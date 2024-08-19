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
	const EigenArrayRGB image_rgb;
	const Eigen::ArrayXXf image, w;
	const int p, p_squared, p_squared_minus_one_div_2, pad, num_threads;
	const Eigen::Index rows, cols, padded_rows, padded_cols;
	const float psnr;

	void create_neighbors(const Eigen::ArrayXXf& array, Eigen::VectorXf& x_, const int i, const int j);
	Eigen::ArrayXXf load_W(const std::string &w_file, const Eigen::Index rows, const Eigen::Index cols);
	Eigen::ArrayXXf compute_custom_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded);
	Eigen::ArrayXXf compute_prediction_error_mask(const Eigen::ArrayXXf& padded_image, Eigen::ArrayXXf& error_sequence, Eigen::VectorXf& coefficients, const bool mask_needed);
	Eigen::ArrayXXf calculate_error_sequence(const Eigen::ArrayXXf& padded, const Eigen::VectorXf& coefficients);

public:
	Watermark(const EigenArrayRGB& image_rgb, const Eigen::ArrayXXf& image, const std::string &w_file_path, const int p, const float psnr);
	EigenArrayRGB make_and_add_watermark(MASK_TYPE type);
	float mask_detector(const Eigen::ArrayXXf& watermarked_image, MASK_TYPE type);
};