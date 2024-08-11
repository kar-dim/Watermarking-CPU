#pragma once

#include <Eigen/Dense>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

enum MASK_TYPE {
	ME,
	NVF
};

using std::string;
using namespace Eigen;

typedef Eigen::Tensor<float, 3> Tensor3d;

#define ME_MASK_CALCULATION_REQUIRED_NO false
#define ME_MASK_CALCULATION_REQUIRED_YES true

class Watermark {

private:
	const ArrayXXf image, w;
	const Tensor3d image_rgb;
	const int p, p_squared, p_squared_minus_one_div_2, pad, num_threads;
	const float psnr;
	const Index rows, cols, elems, padded_cols, padded_rows;

	void create_neighbors(const ArrayXXf& array,VectorXf& x_, const int i, const int j, const int p, const int p_squared);
	ArrayXXf load_W(const string &w_file, const Index rows, const Index cols);
	void compute_NVF_mask(const ArrayXXf& image, const ArrayXXf& padded, ArrayXXf& m_nvf);
	void compute_prediction_error_mask(const ArrayXXf& padded_image, ArrayXXf& m,ArrayXXf& error_sequence, VectorXf& coefficients, const bool mask_needed);
	void compute_error_sequence(const ArrayXXf& padded, const VectorXf& coefficients, ArrayXXf& error_sequence);

public:
	Watermark(const Tensor3d& image_rgb, const ArrayXXf& image, const string &w_file_path, const int p, const float psnr);
	Tensor3d make_and_add_watermark(MASK_TYPE type);
	float mask_detector(const ArrayXXf& watermarked_image, MASK_TYPE type);
};