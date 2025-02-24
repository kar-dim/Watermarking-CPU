#define cimg_use_png
#define cimg_use_jpeg
#include "eigen_rgb_array.hpp"
#include "main_utils.hpp"
#include "Utilities.hpp"
#include "Watermark.hpp"
#include <CImg.h>
#include <cstdint>
#include <cstring>
#include <Eigen/Dense>
#include <exception>
#include <functional>
#include <INIReader.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <omp.h>
#include <sstream>
#include <string>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#define R_WEIGHT 0.299f
#define G_WEIGHT 0.587f
#define B_WEIGHT 0.114f

using namespace cimg_library;
using namespace Eigen;
using std::cout;
using std::string;
using AVPacketPtr = std::unique_ptr<AVPacket, std::function<void(AVPacket*)>>;
using AVFramePtr = std::unique_ptr<AVFrame, std::function<void(AVFrame*)>>;
using AVCodecContextPtr = std::unique_ptr<AVCodecContext, std::function<void(AVCodecContext*)>>;
using FILEPtr = std::unique_ptr<FILE, decltype(&_pclose)>;

/*!
 *  \brief  This is a project implementation of my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU
 *  \author Dimitris Karatzas
 */
int main(int argc, char** argv)
{
	const INIReader inir("settings.ini");
	if (inir.ParseError() < 0) 
	{
		cout << "Could not load configuration file, exiting..";
		exitProgram(EXIT_FAILURE);
	}
	// load parameters
	const int p = inir.GetInteger("parameters", "p", -1);
	const float psnr = inir.GetFloat("parameters", "psnr", -1.0f);
	int numThreads = inir.GetInteger("parameters", "threads", 0);
	if (numThreads <= 0)
	{
		auto threadsSupported = std::thread::hardware_concurrency();
		numThreads = threadsSupported == 0 ? 2 : threadsSupported;
	}

	//openmp initialization
	omp_set_num_threads(numThreads);
#pragma omp parallel for
	for (int i = 0; i < 24; i++) {}

	//check valid parameter values
	if (p <= 1 || p % 2 != 1 || p > 9) 
	{
		cout << "p parameter must be a positive odd number greater than or equal to 3 and less than or equal to 9\n";
		exitProgram(EXIT_FAILURE);
	}
	if (psnr <= 0) 
	{
		cout << "PSNR must be a positive number\n";
		exitProgram(EXIT_FAILURE);
	}

	cout << "Using " << numThreads << " parallel threads.\n";

	//test algorithms
	try {
		const string videoFile = inir.Get("paths", "video", "");
		const int code = videoFile != "" ?
			testForVideo(videoFile, inir, p, psnr) :
			testForImage(inir, p, psnr);
		exitProgram(code);
	}
	catch (const std::exception& ex) {
		cout << ex.what() << "\n";
		exitProgram(EXIT_FAILURE);
	}
	exitProgram(EXIT_SUCCESS);
}

int testForImage(const INIReader& inir, const int p, const float psnr)
{
	const string imagePath = inir.Get("paths", "image", "NO_IMAGE");
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	int loops = inir.GetInteger("parameters", "loops_for_test", 5);
	loops = loops <= 0 || loops > 64 ? 5 : loops;

	//load image from disk
	timer::start();
	const CImg<float> rgbImageCimg(imagePath.c_str());
	timer::end();
	const int rows = rgbImageCimg.height();
	const int cols = rgbImageCimg.width();

	cout << "Each test will be executed " << loops << " times. Average time will be shown below\n";
	cout << "Image size is: " << rows << " rows and " << cols << " columns\n\n";

	//copy from cimg to Eigen
	double secs = timer::elapsedSeconds();
	timer::start();
	const EigenArrayRGB arrayRgb = cimgToEigen3dArray(rgbImageCimg);
	const ArrayXXf arrayGrayscale = eigen3dArrayToGrayscaleArray(arrayRgb, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	timer::end();
	cout << "Time to load image from disk and initialize CImg and Eigen memory objects: " << secs + timer::elapsedSeconds() << " seconds\n\n";
	if (cols <= 16 || rows <= 16 || rows >= 16384 || cols >= 16384)
	{
		cout << "Image dimensions too low or too high\n";
		exitProgram(EXIT_FAILURE);
	}
	//initialize main class responsible for watermarking and detection
	Watermark watermarkObj(rows, cols, inir.Get("paths", "watermark", "w.txt"), p, psnr);
	float watermarkStrength;

	secs = 0;
	//NVF mask calculation
	EigenArrayRGB watermarkNVF, watermarkME;
	for (int i = 0; i < loops; i++)
	{
		timer::start();
		watermarkNVF = watermarkObj.makeWatermark(arrayGrayscale, arrayRgb, watermarkStrength, MASK_TYPE::NVF);
		timer::end();
		secs += timer::elapsedSeconds();
	}
	cout << "Watermark strength (parameter a): " << watermarkStrength << "\nCalculation of NVF mask with " << rows << " rows and " << cols << " columns and parameters:\np = " << p << "  PSNR(dB) = " << psnr << "\n" << executionTime(showFps, secs / loops) << "\n\n";

	secs = 0;
	//Prediction error mask calculation
	for (int i = 0; i < loops; i++)
	{
		timer::start();
		watermarkME = watermarkObj.makeWatermark(arrayGrayscale, arrayRgb, watermarkStrength, MASK_TYPE::ME);
		timer::end();
		secs += timer::elapsedSeconds();
	}
	cout << "Watermark strength (parameter a): " << watermarkStrength << "\nCalculation of ME mask with " << rows << " rows and " << cols << " columns and parameters:\np = " << p << "  PSNR(dB) = " << psnr << "\n" << executionTime(showFps, secs / loops) << "\n\n";

	const ArrayXXf watermarkedNVFgray = eigen3dArrayToGrayscaleArray(watermarkNVF, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	const ArrayXXf watermarkedMEgray = eigen3dArrayToGrayscaleArray(watermarkME, R_WEIGHT, G_WEIGHT, B_WEIGHT);

	float correlationNvf, correlationMe;
	secs = 0;
	//NVF mask detection
	for (int i = 0; i < loops; i++)
	{
		timer::start();
		correlationNvf = watermarkObj.detectWatermark(watermarkedNVFgray, MASK_TYPE::NVF);
		timer::end();
		secs += timer::elapsedSeconds();
	}
	cout << "Calculation of the watermark correlation (NVF) of an image with " << rows << " rows and " << cols << " columns and parameters:\np = " << p << "  PSNR(dB) = " << psnr << "\n" << executionTime(showFps, secs / loops) << "\n\n";

	secs = 0;
	//Prediction error mask detection
	for (int i = 0; i < loops; i++)
	{
		timer::start();
		correlationMe = watermarkObj.detectWatermark(watermarkedMEgray, MASK_TYPE::ME);
		timer::end();
		secs += timer::elapsedSeconds();
	}
	cout << "Calculation of the watermark correlation (ME) of an image with " << rows << " rows and " << cols << " columns and parameters:\np = " << p << "  PSNR(dB) = " << psnr << "\n" << executionTime(showFps, secs / loops) << "\n\n";

	cout << "Correlation [NVF]: " << std::fixed << std::setprecision(16) << correlationNvf << "\n";
	cout << "Correlation [ME]: " << std::fixed << std::setprecision(16) << correlationMe << "\n";

	//save watermarked images to disk
	if (inir.GetBoolean("options", "save_watermarked_files_to_disk", false))
	{
		cout << "\nSaving watermarked files to disk...\n";
#pragma omp parallel sections 
		{
#pragma omp section
			saveWatermarkedImage(imagePath, "_W_NVF", watermarkNVF, IMAGE_TYPE::PNG);
#pragma omp section
			saveWatermarkedImage(imagePath, "_W_ME", watermarkME, IMAGE_TYPE::PNG);
		}
		cout << "Successully saved to disk\n";
	}
	return 0;
}

//embed watermark for a video or try to detect watermark in a video
int testForVideo(const string& videoFile, const INIReader& inir, const int p, const float psnr)
{
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	const int watermarkInterval = inir.GetInteger("parameters_video", "watermark_interval", 30);

	//Set ffmpeg log level
	av_log_set_level(AV_LOG_INFO);

	//Load input video
	AVFormatContext* inputFormatCtx = nullptr;
	if (avformat_open_input(&inputFormatCtx, videoFile.c_str(), nullptr, nullptr) < 0)
	{
		std::cout << "ERROR: Failed to open input video file\n";
		exitProgram(EXIT_FAILURE);
	}
	avformat_find_stream_info(inputFormatCtx, nullptr);
	av_dump_format(inputFormatCtx, 0, videoFile.c_str(), 0);

	//Find video stream and open video decoder
	const int videoStreamIndex = findVideoStreamIndex(inputFormatCtx);
	const AVCodecContextPtr inputDecoderCtx(openDecoderContext(inputFormatCtx->streams[videoStreamIndex]->codecpar), [](AVCodecContext* ctx) { avcodec_free_context(&ctx); });

	//initialize watermark functions class
	const int height = inputFormatCtx->streams[videoStreamIndex]->codecpar->height;
	const int width = inputFormatCtx->streams[videoStreamIndex]->codecpar->width;
	Watermark watermarkObj(height, width, inir.Get("paths", "watermark", ""), p, psnr);

	//realtime watermarking of raw video
	const string makeWatermarkVideoPath = inir.Get("parameters_video", "encode_watermark_file_path", "");
	if (makeWatermarkVideoPath != "")
	{
		const string ffmpegOptions = inir.Get("parameters_video", "encode_options", "-c:v libx265 -preset fast -crf 23");

		// Build the FFmpeg command
		std::ostringstream ffmpegCmd;
		ffmpegCmd << "ffmpeg -y -f rawvideo -pix_fmt yuv420p " << "-s " << width << "x" << height
			<< " -r 30 -i - -i " << videoFile << " " << ffmpegOptions
			<< " -map 0:v -map 1:a -shortest " << makeWatermarkVideoPath;

		// Open FFmpeg process
		FILEPtr ffmpegPipe(_popen(ffmpegCmd.str().c_str(), "wb"), _pclose);
		if (!ffmpegPipe.get())
		{
			std::cout << "Error: Could not open FFmpeg pipe\n";
			exitProgram(EXIT_FAILURE);
		}

		timer::start();
		//read frames
		float watermarkStrength;
		std::unique_ptr<uint8_t> inputFramePtr(new uint8_t[width * height]);
		ArrayXXf inputFrame;
		Array<uint8_t, Dynamic, Dynamic> watermarkedFrame;
		const AVPacketPtr packet(av_packet_alloc(), [](AVPacket* pkt) { av_packet_free(&pkt); });
		const AVFramePtr frame(av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); });
		int framesCount = 0;

		while (av_read_frame(inputFormatCtx, packet.get()) >= 0)
		{
			if (!receivedValidVideoFrame(inputDecoderCtx.get(), packet.get(), frame.get(), videoStreamIndex))
				continue;
			const bool embedWatermark = framesCount % watermarkInterval == 0;
			//if there is row padding (for alignment), we must copy the data to a contiguous block!
			if (frame->linesize[0] != width)
			{
				if (embedWatermark)
				{
					//#pragma omp parallel for //if multi-threaded encoder don't parallelize!
					for (int y = 0; y < height; y++)
						memcpy(inputFramePtr.get() + y * width, frame->data[0] + y * frame->linesize[0], width);

					//TODO check optimizations (RowMajor for watermarkedFrame to not use transpose etc)
					inputFrame = Map<Array<uint8_t, Dynamic, Dynamic>>(inputFramePtr.get(), width, height).transpose().cast<float>();
					watermarkedFrame = watermarkObj.makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).transpose().cast<uint8_t>();
				}
				//write from pinned memory directly (plus UV planes)
				for (int y = 0; y < height; y++)
					fwrite((embedWatermark ? watermarkedFrame.data() + y * width : frame->data[0] + y * frame->linesize[0]), 1, width, ffmpegPipe.get());
				for (int y = 0; y < height / 2; y++)
					fwrite(frame->data[1] + y * frame->linesize[1], 1, width / 2, ffmpegPipe.get());
				for (int y = 0; y < height / 2; y++)
					fwrite(frame->data[2] + y * frame->linesize[2], 1, width / 2, ffmpegPipe.get());

			}
			//else, use original pointer, no need to copy data to intermediate pinned buffer
			else
			{
				if (embedWatermark)
				{
					inputFrame = Map<Array<uint8_t, Dynamic, Dynamic>>(frame->data[0], width, height).transpose().cast<float>();
					watermarkedFrame = watermarkObj.makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).transpose().cast<uint8_t>();
				}
				// Write modified frame to ffmpeg (pipe)
				fwrite(embedWatermark ? watermarkedFrame.data() : frame->data[0], 1, width * frame->height, ffmpegPipe.get());
				fwrite(frame->data[1], 1, width * frame->height / 4, ffmpegPipe.get());
				fwrite(frame->data[2], 1, width * frame->height / 4, ffmpegPipe.get());
			}

			framesCount++;
			av_packet_unref(packet.get());
		}
		timer::end();
		cout << "\nWatermark embeding total execution time: " << executionTime(false, timer::elapsedSeconds()) << "\n";

		//clReleaseMemObject(pinnedBuff);
	}

	//realtime watermarked video detection
	else if (inir.GetBoolean("parameters_video", "watermark_detection", false))
	{
		timer::start();
		float correlation;
		ArrayXXf inputFrame;
		std::unique_ptr<uint8_t> inputFramePtr(new uint8_t[width * height]);
		const AVPacketPtr packet(av_packet_alloc(), [](AVPacket* pkt) { av_packet_free(&pkt); });
		const AVFramePtr frame(av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); });
		int framesCount = 0;

		while (av_read_frame(inputFormatCtx, packet.get()) >= 0)
		{
			if (!receivedValidVideoFrame(inputDecoderCtx.get(), packet.get(), frame.get(), videoStreamIndex))
				continue;

			//detect watermark after X frames
			if (framesCount % watermarkInterval == 0)
			{
				//if there is row padding (for alignment), we must copy the data to a contiguous block!
				const bool rowPadding = frame->linesize[0] != width;
				if (rowPadding)
				{
					#pragma omp parallel for
					for (int y = 0; y < height; y++)
						memcpy(inputFramePtr.get() + y * width, frame->data[0] + y * frame->linesize[0], width);
				}
				//supply the input frame to the GPU and run the detection of the watermark
				inputFrame = Map<Array<uint8_t, Dynamic, Dynamic>>(rowPadding ? inputFramePtr.get() : frame->data[0], width, height).transpose().cast<float>();
				correlation = watermarkObj.detectWatermark(inputFrame, MASK_TYPE::ME);
				cout << "Correlation for frame: " << framesCount << ": " << correlation << "\n";
			}
			framesCount++;
			av_packet_unref(packet.get());
		}
		timer::end();
		cout << "\nWatermark detection total execution time: " << executionTime(false, timer::elapsedSeconds()) << "\n";
		cout << "\nWatermark detection average execution time per frame: " << executionTime(showFps, timer::elapsedSeconds() / framesCount) << "\n";
	}

	// Cleanup
	avformat_close_input(&inputFormatCtx);
	return EXIT_SUCCESS;
}

int findVideoStreamIndex(const AVFormatContext* inputFormatCtx)
{
	int videoStreamIndex = -1;
	for (unsigned int i = 0; i < inputFormatCtx->nb_streams; i++)
	{
		if (inputFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
		{
			videoStreamIndex = i;
			break;
		}
	}
	if (videoStreamIndex == -1)
	{
		std::cout << "ERROR: No video stream found\n";
		exitProgram(EXIT_FAILURE);
	}
	return videoStreamIndex;
}

//open decoder context for video
AVCodecContext* openDecoderContext(const AVCodecParameters* inputCodecParams)
{
	const AVCodec* inputDecoder = avcodec_find_decoder(inputCodecParams->codec_id);
	AVCodecContext* inputDecoderCtx = avcodec_alloc_context3(inputDecoder);
	avcodec_parameters_to_context(inputDecoderCtx, inputCodecParams);
	avcodec_open2(inputDecoderCtx, inputDecoder, nullptr);
	return inputDecoderCtx;
}

//supply a packet to the decoder and check if the received frame is valid by checking its format
bool receivedValidVideoFrame(AVCodecContext* inputDecoderCtx, AVPacket* packet, AVFrame* frame, const int videoStreamIndex)
{
	if (packet->stream_index != videoStreamIndex)
	{
		av_packet_unref(packet);
		return false;
	}
	avcodec_send_packet(inputDecoderCtx, packet);
	if (avcodec_receive_frame(inputDecoderCtx, frame) != 0)
		return false;
	return frame->format == AV_PIX_FMT_YUV420P;
}

//calculate execution time in seconds, or show FPS value
string executionTime(const bool showFps, const double seconds) 
{
	std::stringstream ss;
	if (showFps)
		ss << "FPS: " << std::fixed << std::setprecision(2) << 1.0 / seconds << " FPS";
	else
		ss << std::fixed << std::setprecision(6) << seconds << " seconds";
	return ss.str();
}

//save the provided Eigen RGB array containing a watermarked image to disk
void saveWatermarkedImage(const string& imagePath, const string& suffix, const EigenArrayRGB& watermark, const IMAGE_TYPE type)
{
	const string watermarkedFile = addSuffixBeforeExtension(imagePath, suffix);
	type == IMAGE_TYPE::PNG ? eigen3dArrayToCimg(watermark).save_png(watermarkedFile.c_str())
							: eigen3dArrayToCimg(watermark).save_jpeg(watermarkedFile.c_str(), 100);
}

//exits the program with the provided exit code
void exitProgram(const int exitCode) 
{
	std::system("pause");
	std::exit(exitCode);
}