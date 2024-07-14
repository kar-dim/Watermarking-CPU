#pragma once

#include <Eigen/Dense>
#include <string>

enum MASK_TYPE {
	ME,
	NVF
};

#define MASK_CALCULATION_REQUIRED_NO false
#define MASK_CALCULATION_REQUIRED_YES true

class WatermarkFunctions {
public:
	WatermarkFunctions(const Eigen::ArrayXXf& image, const std::string w_file_path, const int p, const float psnr, const int num_threads);
	Eigen::ArrayXXf make_and_add_watermark_NVF();
	Eigen::ArrayXXf make_and_add_watermark_prediction_error();
	float mask_detector(const Eigen::ArrayXXf& watermarked_image, MASK_TYPE type);
private:
	const Eigen::ArrayXXf image, w;
	const int p, p_squared, p_squared_minus_one_div_2, pad, num_threads;
	const float psnr;
	const Eigen::Index rows, cols, elems, padded_cols, padded_rows;

	Eigen::VectorXf create_neighbors(const Eigen::ArrayXXf& padded_image, const int i, const int j, const int p, const int p_squared);
	Eigen::ArrayXXf load_W(const std::string w_file, const Eigen::Index rows, const Eigen::Index cols);
	Eigen::ArrayXXf make_and_add_watermark(MASK_TYPE type);
	void compute_NVF_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded, Eigen::ArrayXXf& m_nvf);
	void compute_prediction_error_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded_image, Eigen::ArrayXXf& m, Eigen::ArrayXXf& error_sequence, Eigen::MatrixXf& coefficients, const bool mask_needed);
	void compute_error_sequence(const Eigen::ArrayXXf& padded, Eigen::MatrixXf& coefficients, Eigen::ArrayXXf& error_sequence);
};