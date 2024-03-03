#pragma once

#include <Eigen/Dense>
class WatermarkFunctions {
public:
	WatermarkFunctions(const Eigen::ArrayXXf& image, std::string w_file_path, const int p, const float psnr, const int num_threads);
	Eigen::ArrayXXf make_and_add_watermark_NVF();
	Eigen::ArrayXXf make_and_add_watermark_prediction_error();
	float mask_detector(const Eigen::ArrayXXf& watermarked_image, const bool is_nvf);
private:
	Eigen::ArrayXXf image, w;
	int p, p_squared, pad, num_threads;
	float psnr;
	Eigen::Index rows, cols, elems, padded_cols, padded_rows;

	Eigen::VectorXf create_neighbors(const Eigen::ArrayXXf& padded_image, const int i, const int j, const int p, const int p_squared);
	Eigen::ArrayXXf load_W(std::string w_file, const Eigen::Index rows, const Eigen::Index cols);
	Eigen::ArrayXXf make_and_add_watermark(bool is_custom_mask);
	void compute_NVF_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded, Eigen::ArrayXXf& m_nvf);
	void compute_prediction_error_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded_image, Eigen::ArrayXXf& m, Eigen::ArrayXXf& error_sequence, Eigen::MatrixXf& coefficients, const bool mask_needed);
	void compute_error_sequence(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded, Eigen::MatrixXf& coefficients, Eigen::ArrayXXf& error_sequence);
};