# ICSD thesis Part 1 / CPU Watermarking

![512](https://github.com/user-attachments/assets/02298937-2406-409b-8ed6-32d783ea8710)

Code for my Diploma thesis at Information and Communication Systems Engineering (University of the Aegean, School of Engineering) with title "Efficient implementation of watermark and watermark detection algorithms for image and video using the graphics processing unit". [Link](https://hellanicus.lib.aegean.gr/handle/11610/19672)


# Overview

The aim of this project is to compare the performance (primarily execution speed) of watermarking algorithms when implemented on CPU versus GPU. This repository focuses on the CPU implementation (Part 1). The GPU implementation can be found in the corresponding GPU repository (Part 2 / GPU usage [here](https://github.com/kar-dim/Watermarking-GPU) ).

# Key Features

- Implementation of watermark embedding and detection algorithms for images.
- Comparative performance analysis between CPU and GPU implementations.

# Libraries Used

- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page): A C++ template library for linear algebra.
- [CImg](https://cimg.eu/): A C++ library for image processing.
- [inih](https://github.com/jtilly/inih): A lightweight C++ library for parsing .ini configuration files.

# Additional Dependencies

LibPNG and LibJPEG are also included, and are used internally by CImg for loading and saving of images.
