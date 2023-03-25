#define _CRT_SECURE_NO_WARNINGS
#include "INIReader.h"
#define cimg_use_png
#include "CImg.h"
#include "UtilityFunctions.h"
#include "WatermarkFunctions.h"
#include <random>
#include <iostream>
#include <thread>
#include <omp.h>
#include <Eigen/Dense>
#include <iomanip>
#include <string>
#include <cmath>
using namespace cimg_library;
using std::cout;
using std::string;

int main(int argc, char** argv)
{
	int p, num_threads;
	float psnr;
	string image_path, w_file;
	const char* image_path_c = NULL;
	//διάβασμα των παραμέτρων του αρχείου settings.ini, inih βιβλιοθήκη με c++ bindings
	//τη τροποποίησα ώστε να μπορεί να διαβάσει και uint64 τιμές (για το watermark key)
	INIReader inir("settings.ini");
	if (inir.ParseError() < 0) {
		cout << "Could not load configuration file, attempting to read from command line arguments\n";
		//αν δε μπορέσουμε να διαβάσουμε από ini, προσπαθούμε μέσω command line arguments με την εξής σειρα
		//image_path, p, psnr, w_file, num_threads
		if (argc != 6) {
			cout << "Could not load parameters from command line either, exiting application\n";
			return -1;
		}
		image_path.assign(argv[1]);
		image_path_c = image_path.c_str();
		p = std::stoi(argv[2]);
		psnr = std::stof(argv[3]);
		w_file.assign(argv[4]);
		num_threads = std::stoi(argv[5]);

	}
	else {

		//διάβασμα του path της εικόνας
		image_path = inir.Get("paths", "image", "NO_IMAGE");
		image_path_c = image_path.c_str();

		//διάβασμα παραμέτρων p και psnr
		p = inir.GetInteger("parameters", "p", 5);
		psnr = static_cast<float>(inir.GetReal("parameters", "psnr", 30.0f));

		//διάβασμα του ονόματος του αρχείου που έχει το W πίνακα
		w_file = inir.Get("paths", "w_path", "w.txt");

		//διάβασμα του αριθμού των threads που μπορούν να τρέξουν παράλληλα
		num_threads = inir.GetInteger("parameters", "threads", 0);
	}
	if (num_threads <= 0 || num_threads > 256) {
		//αν δόθηκε μεγάλη τιμή ή αρνητική/μηδέν, τότε δίνουμε τη τιμή που επιστρέφει η c++
		//μόνο αν μπορεί να επιστρέψει κάποια τιμή (δεν είναι 0 που σημαίνει δε μπορεί)
		if (std::thread::hardware_concurrency() != 0)
			num_threads = std::thread::hardware_concurrency();
		//αλλιώς δίνουμε μια μικρή τιμή για συμβατότητα όπως η 2
		else {
			num_threads = 2;
		}
	}
	cout << "Using " << num_threads << " parallel threads.\n";
	//θέτουμε στην OpenMP
	omp_set_num_threads(num_threads);
	//αρχικοποίηση των threads που θα τρέχουν παράλληλα
#pragma omp parallel for
	for (int i = 0; i < 24; i++) {}

	//φόρτωση εικόνας
	float* grayscale_vals = NULL;
	float* data = NULL;
	float* data2 = NULL;
	try {
		CImg<float> rgb_image(image_path_c);
		const int rows = rgb_image.height();
		const int cols = rgb_image.width();
		const int elems = rows * cols;

		if (cols <= 16 || rows <= 16 || rows >= 16384 || cols >= 16384) {
			cout << "Image dimensions too low or too high\n";
			return -1;
		}
		if (p <= 0 || p % 2 != 1 || p > 9) {
			cout << "p parameter must be a positive odd number less than 9\n";
			return -1;
		}
		if (psnr <= 0) {
			cout << "PSNR must be a positive number\n";
			return -1;
		}
		cout << "Image size is: " << rows << " rows and " << cols << " columns\n\n";
		float* image_vals = rgb_image.data();
		grayscale_vals = new float[elems];
		//μετατροπή rgb σε grayscale με τα παραπάνω βάρη
		for (int i = 0; i < elems; i++) {
			grayscale_vals[i] = static_cast<float>(std::round(0.299 * image_vals[i]) + std::round(0.587 * image_vals[i + elems]) + std::round(0.114 * image_vals[i + 2 * elems]));
		}
		//διαβάζουμε τον W πίνακα
		Eigen::ArrayXXf w = load_W(w_file, rows, cols);

		//τώρα έχουμε την εικόνα σε float τιμές, δημιουργούμε τον Eigen πίνακα που θα κρατάει τις τιμές
		Eigen::ArrayXXf image_m = Eigen::Map<Eigen::ArrayXXf>(grayscale_vals, cols, rows);
		//transpose διότι διαβάζει COLUMN-WISE η Eigen
		image_m.transposeInPlace();

		//εδώ καλούμε τη συνάρτηση που υπολογίζει την NVF mask και την ενθέτει
		double secs = 0;
		int V = 5;
		Eigen::ArrayXXf image_m_nvf, image_m_me;
		for (int i = 0; i < V; i++) {
			timer::start();
			image_m_nvf = make_and_add_watermark_NVF(image_m, w, p, psnr, num_threads);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate NVF mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / V << " seconds.\n\n";

		//αντίστοιχα για τη ME mask
		secs = 0;
		for (int i = 0; i < V; i++) {
			timer::start();
			image_m_me = make_and_add_watermark_ME(image_m, w, p, psnr, num_threads);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate ME mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / V << " seconds.\n\n";

		float correlation_nvf, correlation_me;
		secs = 0;
		for (int i = 0; i < V; i++) {
			timer::start();
			correlation_nvf = mask_detector(image_m_nvf, w, p, psnr, num_threads, true);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate NVF [COR] of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / V << " seconds.\n\n";

		secs = 0;
		for (int i = 0; i < V; i++) {
			timer::start();
			correlation_me = mask_detector(image_m_me, w, p, psnr, num_threads, false);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << "Time to calculate ME [COR] of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << secs / V << " seconds.\n\n";

		//correlation μεταξύ των τελικών υδατογραφημένων εικόνων, για NVF και ΜΕ
		cout << "Correlation [NVF]: " << std::fixed << std::setprecision(16) << correlation_nvf << "\n";
		cout << "Correlation [ME]: " << std::fixed << std::setprecision(16) << correlation_me << "\n";


		//αποδέσμευση μνήμης
		delete[] data;
		delete[] grayscale_vals;
		delete[] data2;
		data = NULL;
		data2 = NULL;
		grayscale_vals = NULL;

		//system("pause");
		return 0;
	}
	catch (...) {
		delete[] grayscale_vals;
		delete[] data;
		delete[] data2;
		return 0;
	}
}