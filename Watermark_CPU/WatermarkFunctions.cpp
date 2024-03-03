#include "WatermarkFunctions.h"
#include "UtilityFunctions.h"
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <vector>
#include <memory>
#include <Eigen/Dense>

using std::cout;

WatermarkFunctions::WatermarkFunctions(const Eigen::ArrayXXf& image, std::string w_file_path, const int p, const float psnr, const int num_threads) {
	this->image = image;
	this->p = p;
	this->pad = p / 2;
	this->rows = image.rows();
	this->cols = image.cols();
	this->padded_rows = rows + 2 * pad;
	this->padded_cols = cols + 2 * pad;
	this->elems = rows * cols;
	this->w = load_W(w_file_path, rows, cols);
	this->p_squared = static_cast<int>(std::pow(p, 2));
	this->psnr = psnr;
	this->num_threads = num_threads;
}
//συνάρτηση που διαβάζει τον W πίνακα και τον επιστρέφει σε Eigen/Array μορφή
Eigen::ArrayXXf WatermarkFunctions::load_W(std::string w_file, const Eigen::Index rows, const Eigen::Index cols) {
	int i = 0;
	std::ifstream w_stream;

	w_stream.open(w_file.c_str(), std::ios::binary);
	if (!w_stream.is_open()) {
		std::string error_str("Error opening '" + w_file + "' file for Random noise W array");
		cout << error_str;
		throw std::exception(error_str.c_str());
	}
	auto w_ptr = std::unique_ptr<float>(new float[rows * cols]);
	while (!w_stream.eof())
		w_stream.read(reinterpret_cast<char*>(&w_ptr.get()[i++]), sizeof(float));
	Eigen::ArrayXXf w = Eigen::Map<Eigen::ArrayXXf>(w_ptr.get(), cols, rows);
	w.transposeInPlace();
	return w;
}

//generate p x p neighbors
Eigen::VectorXf WatermarkFunctions::create_neighbors(const Eigen::ArrayXXf& padded_image, const int i, const int j, const int p, const int p_squared)
{
	const int middle = static_cast<int>(floor(p_squared / 2));
	Eigen::ArrayXXf x_temp(p, p);
	//x_: will contain all the neighbors minus the current pixel value
	Eigen::VectorXf x_(p_squared - 1);
	int i0, i1, j0, j1;
	const float* x_temp_ptr = x_temp.data();
	i0 = i - (p - 1) / 2;
	j0 = j - (p - 1) / 2;
	i1 = i + (p - 1) / 2;
	j1 = j + (p - 1) / 2;
	x_temp = padded_image.block(i0, j0, i1 - i0 + 1, j1 - j0 + 1);
	//ignore the central pixel value
	int k = 0;
	for (int i = 0; i < x_temp.size(); i++) {
		if (i != middle) {
			x_(k) = x_temp_ptr[i];
			k++;
		}
	}
	return x_;
}
void WatermarkFunctions::compute_NVF_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded, Eigen::ArrayXXf& m_nvf)
{
	m_nvf = Eigen::ArrayXXf::Constant(rows, cols, 0.0f);
#pragma omp parallel for
	for (int i = pad; i < rows + pad; i++) {
		for (int j = pad; j < cols + pad; j++) {
			Eigen::ArrayXXf neighb = Eigen::ArrayXXf::Constant(p, p, 0.0f);
			int i0, i1, j0, j1;
			float mean = 0.0f, variance = 0.0f;
			const float* neighb_d = neighb.data();
			i0 = i - (p - 1) / 2;
			j0 = j - (p - 1) / 2;
			i1 = i + (p - 1) / 2;
			j1 = j + (p - 1) / 2;
			neighb = padded.block(i0, j0, i1 - i0 + 1, j1 - j0 + 1);
			mean = neighb.mean();
			for (int ii = 0; ii < p; ii++) {
				for (int jj = 0; jj < p; jj++) {
					variance += powf(neighb_d[jj * p + ii] - mean, 2);
				}
			}
			variance = variance / (p_squared - 1);
			m_nvf(i - pad, j - pad) = 1.0f - (1.0f / (1.0f + variance));
			variance = 0.0f;
		}
	}
}

Eigen::ArrayXXf WatermarkFunctions::make_and_add_watermark(bool is_custom_mask) {
	Eigen::ArrayXXf padded = Eigen::ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = image;
	Eigen::ArrayXXf m, u;
	if (is_custom_mask) {
		compute_NVF_mask(image, padded, m);
	}
	else {
		Eigen::ArrayXXf error_sequence;
		Eigen::MatrixXf coefficients;
		compute_ME_mask(image, padded, m, error_sequence, coefficients, true);
	}
	u = m * w;
	float divisor = std::sqrt(u.square().sum() / (rows * cols));
	float a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divisor;
	return image + (a * u).eval();
}

//create NVF mask and return the watermarked image
Eigen::ArrayXXf WatermarkFunctions::make_and_add_watermark_NVF()
{
	return make_and_add_watermark(true);
}

//create ME mask and return the watermarked image
Eigen::ArrayXXf WatermarkFunctions::make_and_add_watermark_ME()
{
	return make_and_add_watermark(false);
}

//compute Prediction error mask
void WatermarkFunctions::compute_ME_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded_image, Eigen::ArrayXXf& m_e, Eigen::ArrayXXf& error_sequence, Eigen::MatrixXf& coefficients, const bool mask_needed)
{
	error_sequence = Eigen::ArrayXXf::Constant(rows, cols, 0.0f);
	m_e = Eigen::ArrayXXf::Constant(rows, cols, 0.0f);
	Eigen::MatrixXf Rx = Eigen::ArrayXXf::Constant(p_squared - 1, p_squared - 1, 0.0f);
	Eigen::MatrixXf rx = Eigen::ArrayXXf::Constant(p_squared - 1, 1, 0.0f);
	std::vector<Eigen::MatrixXf> Rx_all(num_threads);
	std::vector<Eigen::MatrixXf> rx_all(num_threads);
	for (int i = 0; i < num_threads; i++) {
		Rx_all[i] = Rx;
		rx_all[i] = rx;
	}
	double sum = 0;
#pragma omp parallel for
	for (int i = pad; i < rows + pad; i++) {
		Eigen::VectorXf x_;
		for (int j = pad; j < cols + pad; j++) {
			//calculate p^-1 γειτονιά
			x_ = create_neighbors(padded_image, i, j, p, p_squared);
			//calculate Rx and rx
			sum += timer::secs_passed();
			Rx_all[omp_get_thread_num()].noalias() += (x_ * x_.transpose().eval());
			rx_all[omp_get_thread_num()].noalias() += (x_ * image(i - pad, j - pad));
		}
	}
	//reduction sums of Rx,rx of each thread
	for (int i = 0; i < num_threads; i++) {
		Rx.noalias() += Rx_all[i];
		rx.noalias() += rx_all[i];
	}
	coefficients = Rx.fullPivLu().solve(rx);
	coefficients.transposeInPlace();

	//calculate ex(i,j)
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		Eigen::VectorXf x_;
		for (int j = 0; j < cols; j++) {
			x_ = create_neighbors(padded_image, i + pad, j + pad, p, p_squared);
			error_sequence(i, j) = image(i, j) - (coefficients * x_)(0);
		}
	}
	if (mask_needed) {
		Eigen::ArrayXXf error_sequence_abs = error_sequence.abs().eval();
		float ex_max = error_sequence_abs.maxCoeff();
		m_e = error_sequence_abs / ex_max;
	}
}

//computes the prediction error sequence 
void WatermarkFunctions::compute_error_sequence(const Eigen::ArrayXXf& image, Eigen::MatrixXf& coefficients, Eigen::ArrayXXf& error_sequence)
{
	error_sequence = Eigen::ArrayXXf::Constant(rows, cols, 0.0f);
	Eigen::ArrayXXf padded = Eigen::ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = image;
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		Eigen::VectorXf x_;
		for (int j = 0; j < cols; j++) {
			x_ = create_neighbors(padded, i + pad, j + pad, p, p_squared);
			error_sequence(i, j) = image(i, j) - (coefficients * x_)(0);
		}
	}

}
//main mask detector for Me and NVF masks
float WatermarkFunctions::mask_detector(const Eigen::ArrayXXf& watermarked_image, const bool is_custom_mask)
{
	Eigen::ArrayXXf m, e_z, padded = Eigen::ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
	Eigen::MatrixXf a_z;
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = watermarked_image;
	if (is_custom_mask) {
		compute_ME_mask(watermarked_image, padded, m, e_z, a_z, false); //no need for Me mask
		compute_NVF_mask(watermarked_image, padded, m);
	}
	else {
		compute_ME_mask(watermarked_image, padded, m, e_z, a_z, true);
	}

	Eigen::ArrayXXf u = m * w;
	Eigen::ArrayXXf e_u;
	compute_error_sequence(u, a_z, e_u);
	float dot_ez_eu, d_ez, d_eu;
#pragma omp parallel sections
	{
#pragma omp section
		dot_ez_eu = (e_z * e_u).eval().sum();
#pragma omp section
		d_ez = std::sqrt(e_z.square().eval().sum());
#pragma omp section
		d_eu = std::sqrt(e_u.square().eval().sum());
	}
	return dot_ez_eu / (d_ez * d_eu);
}
