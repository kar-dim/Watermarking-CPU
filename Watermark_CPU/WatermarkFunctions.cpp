#include "WatermarkFunctions.h"
#include "UtilityFunctions.h"
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <numeric>
#include <vector>
#include <Eigen/Dense>
#include <CImg.h>

using namespace cimg_library;
using std::cout;

//συνάρτηση που διαβάζει τον W πίνακα και τον επιστρέφει σε Eigen/Array μορφή
Eigen::ArrayXXf load_W(std::string w_file, const int rows, const int cols) {
	float* w_ptr = NULL;
	int i = 0;
	std::ifstream w_stream;

	w_stream.open(w_file.c_str(), std::ios::binary);
	if (!w_stream.is_open()) {
		cout << "\nError opening \"" << w_file << "\" file.";
		system("pause");
		exit(-1);
	}
	w_ptr = new float[rows * cols];
	while (!w_stream.eof()) {
		w_stream.read(reinterpret_cast<char*>(&w_ptr[i]), sizeof(float));
		i++;
	}
	Eigen::ArrayXXf w = Eigen::Map<Eigen::ArrayXXf>(w_ptr, cols, rows);
	w.transposeInPlace();
	delete[] w_ptr;
	return w;
}

//συνάρτηση που παράγει τους p x p γείτονες
Eigen::VectorXf create_neighbors(const Eigen::ArrayXXf& padded_image, const int i, const int j, const int p, const int p_squared)
{
	const int middle = static_cast<int>(floor(p_squared / 2));
	//x_temp: όλη η γειτονιά
	Eigen::ArrayXXf x_temp(p, p);
	//x_: όλη η γειτονιά μείων το κεντρικό σημείο
	Eigen::VectorXf x_(p_squared - 1);
	int i0, i1, j0, j1;
	const float* x_temp_ptr = x_temp.data();
	i0 = i - (p - 1) / 2;
	j0 = j - (p - 1) / 2;
	i1 = i + (p - 1) / 2;
	j1 = j + (p - 1) / 2;
	x_temp = padded_image.block(i0, j0, i1 - i0 + 1, j1 - j0 + 1);
	//το παρακάτω κομμάτι κώδικα δίνει στο x_ μόνο τα στοιχεία της γειτονιάς και όχι το κεντρικό
	int k = 0;
	for (int i = 0; i < x_temp.size(); i++) {
		if (i != middle) {
			x_(k) = x_temp_ptr[i];
			k++;
		}
	}
	return x_;
}
void compute_NVF_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded, Eigen::ArrayXXf& m_nvf, const int p, const int pad, const Eigen::Index rows, Eigen::Index cols)
{
	const auto elems = rows * cols;

	//padding
	const int p_squared = static_cast<int>(std::pow(p, 2));
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;

	//θα τρέξουν παράλληλα για Ν threads τον υπολογισμό της μάσκας. Κάθε thread αναλαμβάνει τον υπολογισμό
	//των (rows*cols)/N pixels, έτσι γίνεται επιτάχυνση της διαδικασίας. Οτιδήποτε ορίζεται μέσα στο omp pragma
	//είναι private μεταβλητή, δηλαδή κάθε thread έχει ξεχωριστό pad block, variance, mean και θέσεις block.
#pragma omp parallel for
	for (int i = pad; i < rows + pad; i++) {
		for (int j = pad; j < cols + pad; j++) {
			Eigen::ArrayXXf neighb = Eigen::ArrayXXf::Constant(p, p, 0.0f);
			int i0, i1, j0, j1;
			float mean = 0.0f, variance = 0.0f;
			//απευθείας διάβασμα μέσω pointer παρα μέσω του () operator για επιτάχυνση
			const float* neighb_d = neighb.data();
			i0 = i - (p - 1) / 2;
			j0 = j - (p - 1) / 2;
			i1 = i + (p - 1) / 2;
			j1 = j + (p - 1) / 2;
			neighb = padded.block(i0, j0, i1 - i0 + 1, j1 - j0 + 1);
			mean = neighb.mean();
			//εδώ διαβάζονται κατα στήλες οι τιμές της padded
			for (int ii = 0; ii < p; ii++) {
				for (int jj = 0; jj < p; jj++) {
					variance += pow(neighb_d[jj * p + ii] - mean, 2); //double to float warning, ignore!
				}
			}
			variance = variance / (p_squared - 1);
			m_nvf(i - pad, j - pad) = 1.0f - (1.0f / (1.0f + variance));
			variance = 0.0f;
		}
	}
}
//συνάρτηση που δημιουργεί τη μάσκα NVF και την ενθέτει στην τελική υδατογραφημένη εικόνα
Eigen::ArrayXXf make_and_add_watermark_NVF(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& w, const int p, const float psnr, const int num_threads)
{
	//για να μην κάνουμε query συνέχεια τις στήλες/γραμμές, τις αποθηκεύουμε
	const auto rows = image.rows();
	const auto cols = image.cols();
	const auto elems = rows * cols;

	//padding
	const int p_squared = static_cast<int>(std::pow(p, 2));
	const int pad = p / 2;
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;
	Eigen::ArrayXXf padded = Eigen::ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
	Eigen::ArrayXXf m_nvf = Eigen::ArrayXXf::Constant(rows, cols, 0.0f);
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = image;

	//υπολογισμός NVF-based μάσκας
	compute_NVF_mask(image, padded, m_nvf, p, pad, rows, cols);

	//η τελική μάσκα u = m*w (point wise πολλαπλασιασμός)
	Eigen::ArrayXXf u = m_nvf * w;

	//υπολογισμός του α παραμέτρου
	float divv = std::sqrt(u.square().sum() / (rows * cols));
	float a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divv;

	//τελικός υπολογισμός της watermarked image
	Eigen::ArrayXXf y = image + (a * u).eval();

	//επιστροφή της υδατογραφημένης εικόνας
	return y;
}

//συνάρτηση που δημιουργεί τη μάσκα ME και την ενθέτει στη τελική υδατογραφημένη εικόνα
Eigen::ArrayXXf make_and_add_watermark_ME(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& w, const int p, const float psnr, const int num_threads)
{
	//για να μην κάνουμε query συνέχεια τις στήλες/γραμμές, τις αποθηκεύουμε
	const auto rows = image.rows();
	const auto cols = image.cols();
	const auto elems = rows * cols;

	//padding
	const int p_squared = static_cast<int>(std::pow(p, 2));
	const int pad = p / 2;
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;
	Eigen::ArrayXXf padded = Eigen::ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = image;

	//υπολογισμός μάσκας
	Eigen::ArrayXXf m_e, e_x;
	Eigen::MatrixXf a_x;
	compute_ME_mask(image, padded, m_e, e_x, a_x, p, pad, rows, cols, num_threads, true);

	//η τελική μάσκα u = m*w (point wise πολλαπλασιασμός)
	Eigen::ArrayXXf u = m_e * w;

	//υπολογισμός του α παραμέτρου
	float divv = std::sqrt(u.square().sum() / (rows * cols));
	float a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divv;

	//τελικός υπολογισμός της watermarked image
	Eigen::ArrayXXf y = image + (a * u).eval();

	//επιστροφή της τελικής υδατογραφημένης εικόνας
	return y;
}

//συνάρτηση που υπολογίζει στην Prediction Error Mask
void compute_ME_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded_image, Eigen::ArrayXXf& m_e, Eigen::ArrayXXf& e_x, Eigen::MatrixXf& a_x, const int p, const int pad, Eigen::Index rows, Eigen::Index cols, const int num_threads, const bool mask_needed)
{
	const auto elems = rows * cols;
	const int p_squared = static_cast<int>(std::pow(p, 2));
	//αρχικοποίηση πινάκων που χρειάζονται στο ME masking
	e_x = Eigen::ArrayXXf::Constant(rows, cols, 0.0f);
	m_e = Eigen::ArrayXXf::Constant(rows, cols, 0.0f);

	//Οι δύο παρακάτω Eigen πίνακες περιέχουν το άθροισμα των Rx,rx κάθε πιξελ
	Eigen::MatrixXf Rx = Eigen::ArrayXXf::Constant(p_squared - 1, p_squared - 1, 0.0f);
	Eigen::MatrixXf rx = Eigen::ArrayXXf::Constant(p_squared - 1, 1, 0.0f);
	//κάθε thread υπολογίζει τα δικά του rx, Rx και x_
	//στο τέλος τα rx και Rx θα αθροιστούν
	std::vector<Eigen::MatrixXf> Rx_all(num_threads);
	std::vector<Eigen::MatrixXf> rx_all(num_threads);
	for (int i = 0; i < num_threads; i++) {
		Rx_all[i] = Rx;
		rx_all[i] = rx;
	}
	double sum = 0;

	//διαδικασία masking, χρήση multi-threading και vectorization, κάθε thread με vectorized add
	//προσθέτει στα δικά του Rx,rx, στο τέλος αθροίζονται για να προκύψουν οι τελικοί Rx,rx πίνακες
#pragma omp parallel for
	for (int i = pad; i < rows + pad; i++) {

		//κάθε thread θα έχει δικία του μεταβλητή γειτονιάς, αλλιώς θα είχαμε πολλά constructions/destructions
		//αν η δήλωση ήταν στο διπλό for μεσα, ενώ θα χρειαζόμασταν locks αν η δήλωση ήταν εκτός for loops
		//Έχωντας τη δήλωση μέσα στο πρώτο for loop το construction/destruction θα γίνει μόνο μια φορά
		Eigen::VectorXf x_;
		for (int j = pad; j < cols + pad; j++) {
			//πρώτα υπολογίζουμε τη p^-1 γειτονιά
			x_ = create_neighbors(padded_image, i, j, p, p_squared);
			//υπολογισμός Rx και rx για το συγκεκριμένο πιξελ
			sum += timer::secs_passed();
			Rx_all[omp_get_thread_num()].noalias() += (x_ * x_.transpose().eval());
			rx_all[omp_get_thread_num()].noalias() += (x_ * image(i - pad, j - pad));
		}
	}
	//τελικό άθροισμα των επιμέρους Rx,rx κάθε thread
	for (int i = 0; i < num_threads; i++) {
		Rx.noalias() += Rx_all[i];
		rx.noalias() += rx_all[i];
	}

	//λύνουμε το σύστημα, επειδή ο πίνακας Rx είναι square, λύνουμε με τη μέθοδο LU decomposition
	a_x = Rx.fullPivLu().solve(rx);
	a_x.transposeInPlace();

	//υπολογισμός της ex(i,j) παράλληλα, αφού κάθε υπολογισμός είναι ανεξάρτητος
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		//αντίστοιχα με τη περίπτωση υπολογισμού των rx/Rx, κάθε thread έχει δικια του μεταβλητή γειτονιας
		Eigen::VectorXf x_;
		for (int j = 0; j < cols; j++) {
			x_ = create_neighbors(padded_image, i + pad, j + pad, p, p_squared);
			//βάζω το (0) για να πάρουμε το πρώτο στοιχείο που είναι και το μοναδικό, απλώς η Eigen
			//δεν επιτρέπεται scalar πλην πίνακας, πρακτικά είναι scalar πλην 1x1πίνακα, δηλαδή scalar πλην scalar
			e_x(i, j) = image(i, j) - (a_x * x_)(0);
		}
	}
	if (mask_needed) {
		//εύρεση της max τιμής σφάλματος για να βρούμε τη μάσκα
		Eigen::ArrayXXf e_x_abs = e_x.abs().eval();
		float ex_max = e_x_abs.maxCoeff();
		m_e = e_x_abs / ex_max;
	}
}

//συνάρτηση που υπολογίζει με έτοιμο φίλτρο (a_x) το e_x
void compute_error_sequence(const Eigen::ArrayXXf& image, Eigen::MatrixXf& a_x, Eigen::ArrayXXf& e_x, const Eigen::Index rows, const Eigen::Index cols, const int p, const int pad)
{
	e_x = Eigen::ArrayXXf::Constant(rows, cols, 0.0f);
	const int p_squared = static_cast<int>(std::pow(p, 2));
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;
	Eigen::ArrayXXf padded = Eigen::ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = image;
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		//αντίστοιχα με τη περίπτωση υπολογισμού των rx/Rx, κάθε thread έχει δικια του μεταβλητή γειτονιας
		Eigen::VectorXf x_;
		for (int j = 0; j < cols; j++) {
			x_ = create_neighbors(padded, i + pad, j + pad, p, p_squared);
			//βάζω το (0) για να πάρουμε το πρώτο στοιχείο που είναι και το μοναδικό, απλώς η Eigen
			//δεν επιτρέπεται scalar πλην πίνακας, πρακτικά είναι scalar πλην 1x1πίνακα, δηλαδή scalar πλην scalar
			e_x(i, j) = image(i, j) - (a_x * x_)(0);
		}
	}

}
//συνάρτηση που υλοποιεί τον watermark detector
float mask_detector(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& w, const int p, const float psnr, const int num_threads, const bool is_nvf)
{
	//για να μην κάνουμε query συνέχεια τις στήλες/γραμμές, τις αποθηκεύουμε
	const auto rows = image.rows();
	const auto cols = image.cols();
	const auto elems = rows * cols;

	//padding
	const int p_squared = static_cast<int>(std::pow(p, 2));
	const int pad = p / 2;
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;


	Eigen::ArrayXXf m, e_z;
	Eigen::MatrixXf a_z;
	if (is_nvf) {
		Eigen::ArrayXXf padded = Eigen::ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
		padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = image;
		compute_ME_mask(image, padded, m, e_z, a_z, p, pad, rows, cols, num_threads, false); //δε χρειάζεται υπολογισμός Me mask
		compute_NVF_mask(image, padded, m, p, pad, rows, cols);
	}
	else {
		Eigen::ArrayXXf padded = Eigen::ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
		padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = image;
		compute_ME_mask(image, padded, m, e_z, a_z, p, pad, rows, cols, num_threads, true);
	}

	//η τελική μάσκα u = m*w (point wise πολλαπλασιασμός)
	Eigen::ArrayXXf u = m * w;

	//μένει να υπολογίσουμε το eu
	Eigen::ArrayXXf padded = Eigen::ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = u;
	//η μάσκα m_eu δε χρειάζεται για τον υπολογισμό του correlation αλλά το error_u
	//εφαρμόζεται ο γρήγορος υπολογισμός με χρήση του έτοιμου φίλτρου a_z
	Eigen::ArrayXXf e_u;
	compute_error_sequence(u, a_z, e_u, rows, cols, p, pad);

	//υπολογισμός correlation, παράλληλα υπολογισμός των d_ez, d_eu και dot_ez_eu
	float dot_ez_eu, d_ez, d_eu, correlation;
#pragma omp parallel sections
	{
#pragma omp section
		dot_ez_eu = (e_z * e_u).eval().sum();
#pragma omp section
		d_ez = std::sqrt(e_z.square().eval().sum());
#pragma omp section
		d_eu = std::sqrt(e_u.square().eval().sum());
	}
	correlation = dot_ez_eu / (d_ez * d_eu);
	return correlation;
}
