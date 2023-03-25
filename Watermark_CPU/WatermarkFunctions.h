#pragma once

#include <Eigen/Dense>
#include <vector>

Eigen::VectorXf create_neighbors(const Eigen::ArrayXXf& padded_image, const int i, const int j, const int p, const int p_squared);
Eigen::ArrayXXf load_W(std::string w_file, const int rows, const int cols);
Eigen::ArrayXXf make_and_add_watermark_NVF(const Eigen::ArrayXXf&image, const Eigen::ArrayXXf &w, const int p, const float psnr, const int num_threads);
Eigen::ArrayXXf make_and_add_watermark_ME(const Eigen::ArrayXXf&image, const Eigen::ArrayXXf& w, const int p, const float psnr, const int num_threads);
void compute_ME_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded_image, Eigen::ArrayXXf& m, Eigen::ArrayXXf& e_x, Eigen::MatrixXf& a_x, const int p, const int pad, const Eigen::Index rows, const Eigen::Index cols, const int num_threads, const bool mask_needed);
void compute_error_sequence(const Eigen::ArrayXXf& image, Eigen::MatrixXf& a_x, Eigen::ArrayXXf& e_x, const Eigen::Index rows, const Eigen::Index cols, const int p, const int pad);
float mask_detector(const Eigen::ArrayXXf& img, const Eigen::ArrayXXf& w, const int p, const float psnr, const int num_threads, const bool is_nvf);