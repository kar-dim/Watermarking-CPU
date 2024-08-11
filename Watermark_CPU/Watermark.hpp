#pragma once

#include <Eigen/Dense>
#include <string>

enum MASK_TYPE {
	ME,
	NVF
};

using std::string;

#define ME_MASK_CALCULATION_REQUIRED_NO false
#define ME_MASK_CALCULATION_REQUIRED_YES true

class WatermarkFunctions {

private:
	const Eigen::ArrayXXf image, w;
	const int p, p_squared, p_squared_minus_one_div_2, pad, num_threads;
	const float psnr;
	const Eigen::Index rows, cols, elems, padded_cols, padded_rows;

	void create_neighbors(const Eigen::ArrayXXf& array, Eigen::VectorXf& x_, const int i, const int j, const int p, const int p_squared);
	Eigen::ArrayXXf load_W(const string &w_file, const Eigen::Index rows, const Eigen::Index cols);
	void compute_NVF_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded, Eigen::ArrayXXf& m_nvf);
	void compute_prediction_error_mask(const Eigen::ArrayXXf& padded_image, Eigen::ArrayXXf& m, Eigen::ArrayXXf& error_sequence, Eigen::VectorXf& coefficients, const bool mask_needed);
	void compute_error_sequence(const Eigen::ArrayXXf& padded, const Eigen::VectorXf& coefficients, Eigen::ArrayXXf& error_sequence);

public:
	WatermarkFunctions(const Eigen::ArrayXXf& image, const string &w_file_path, const int p, const float psnr);
	Eigen::ArrayXXf make_and_add_watermark(MASK_TYPE type);
	float mask_detector(const Eigen::ArrayXXf& watermarked_image, MASK_TYPE type);
};