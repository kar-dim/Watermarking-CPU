#define cimg_use_png
#define cimg_use_jpeg
#include "eigen_rgb_array.hpp"
#include "main_utils.hpp"
#include "Utilities.hpp"
#include "videoprocessingcontext.hpp"
#include "Watermark.hpp"
#include <CImg.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Eigen/Dense>
#include <exception>
#include <format>
#include <functional>
#include <INIReader.h>
#include <iostream>
#include <memory>
#include <omp.h>
#include <sstream>
#include <string>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/packet.h>
#include <libavutil/frame.h>
#include <libavutil/log.h>
#include <libavutil/avutil.h>
#include <libavcodec/codec.h>
#include <libavutil/pixfmt.h>
}

using namespace cimg_library;
using namespace Eigen;
using std::cout;
using std::string;
using AVPacketPtr = std::unique_ptr<AVPacket, std::function<void(AVPacket*)>>;
using AVFramePtr = std::unique_ptr<AVFrame, std::function<void(AVFrame*)>>;
using AVFormatContextPtr = std::unique_ptr<AVFormatContext, std::function<void(AVFormatContext*)>>;
using AVCodecContextPtr = std::unique_ptr<AVCodecContext, std::function<void(AVCodecContext*)>>;
using FILEPtr = std::unique_ptr<FILE, decltype(&_pclose)>;
using ArrayXXu8 = Array<uint8_t, Dynamic, Dynamic>;

//helper lambda function that displays an error message and exits the program if an error condition is true
auto checkError = [](auto criticalErrorCondition, const std::string& errorMessage)
{
	if (criticalErrorCondition)
	{
		std::cout << errorMessage << "\n";
		exitProgram(EXIT_FAILURE);
	}
};

/*!
 *  \brief  This is a project implementation of my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU
 *  \author Dimitris Karatzas
 */
int main(int argc, char** argv)
{
	const INIReader inir("settings.ini");
	checkError(inir.ParseError() < 0, "Could not load settings.ini file");

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
	checkError(p <= 1 || p % 2 != 1 || p > 9, "p parameter must be a positive odd number greater than or equal to 3 and less than or equal to 9");
	checkError(psnr <= 0, "PSNR must be a positive number");

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
	constexpr float rPercent = 0.299f;
	constexpr float gPercent = 0.587f;
	constexpr float bPercent = 0.114f;
	const string imagePath = inir.Get("paths", "image", "NO_IMAGE");
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	int loops = inir.GetInteger("parameters", "loops_for_test", 5);
	loops = loops <= 0 ? 5 : loops;

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
	const ArrayXXf arrayGrayscale = eigen3dArrayToGrayscaleArray(arrayRgb, rPercent, gPercent, bPercent);
	timer::end();
	cout << "Time to load image from disk and initialize CImg and Eigen memory objects: " << secs + timer::elapsedSeconds() << " seconds\n\n";

	checkError(cols <= 16 || rows <= 16 || rows >= 16384 || cols >= 16384, "Image dimensions too low or too high");
	
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
	cout << std::format("Watermark strength(parameter a) : {}\nCalculation of NVF mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", watermarkStrength, rows, cols, p, psnr, executionTime(showFps, secs / loops));

	secs = 0;
	//Prediction error mask calculation
	for (int i = 0; i < loops; i++)
	{
		timer::start();
		watermarkME = watermarkObj.makeWatermark(arrayGrayscale, arrayRgb, watermarkStrength, MASK_TYPE::ME);
		timer::end();
		secs += timer::elapsedSeconds();
	}
	cout << std::format("Watermark strength(parameter a) : {}\nCalculation of ME mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", watermarkStrength, rows, cols, p, psnr, executionTime(showFps, secs / loops));

	const ArrayXXf watermarkedNVFgray = eigen3dArrayToGrayscaleArray(watermarkNVF, rPercent, gPercent, bPercent);
	const ArrayXXf watermarkedMEgray = eigen3dArrayToGrayscaleArray(watermarkME, rPercent, gPercent, bPercent);

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
	cout << std::format("Calculation of the watermark correlation (NVF) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, executionTime(showFps, secs / loops));

	secs = 0;
	//Prediction error mask detection
	for (int i = 0; i < loops; i++)
	{
		timer::start();
		correlationMe = watermarkObj.detectWatermark(watermarkedMEgray, MASK_TYPE::ME);
		timer::end();
		secs += timer::elapsedSeconds();
	}
	cout << std::format("Calculation of the watermark correlation (ME) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, executionTime(showFps, secs / loops));

	cout << std::format("Correlation [NVF]: {:.16f}\n", correlationNvf);
	cout << std::format("Correlation [ME]: {:.16f}\n", correlationMe);

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
	AVFormatContext* rawInputCtx = nullptr;
	checkError(avformat_open_input(&rawInputCtx, videoFile.c_str(), nullptr, nullptr) < 0, "ERROR: Failed to open input video file");
	AVFormatContextPtr inputFormatCtx(rawInputCtx, [](AVFormatContext* ctx) { if (ctx) { avformat_close_input(&ctx); } });
	avformat_find_stream_info(inputFormatCtx.get(), nullptr);
	av_dump_format(inputFormatCtx.get(), 0, videoFile.c_str(), 0);

	//Find video stream and open video decoder
	const int videoStreamIndex = findVideoStreamIndex(inputFormatCtx.get());
	checkError(videoStreamIndex == -1, "ERROR: No video stream found");
	const AVCodecContextPtr inputDecoderCtx(openDecoderContext(inputFormatCtx->streams[videoStreamIndex]->codecpar), [](AVCodecContext* ctx) { avcodec_free_context(&ctx); });

	//initialize watermark functions class
	const int height = inputFormatCtx->streams[videoStreamIndex]->codecpar->height;
	const int width = inputFormatCtx->streams[videoStreamIndex]->codecpar->width;
	Watermark watermarkObj(height, width, inir.Get("paths", "watermark", ""), p, psnr);

	//initialize necessary FFmpeg structures (packet, frame)
	const AVPacketPtr packet(av_packet_alloc(), [](AVPacket* pkt) { av_packet_free(&pkt); });
	const AVFramePtr frame(av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); });
	std::unique_ptr<uint8_t> inputFramePtr(new uint8_t[width * height]);

	//group common video data for both embedding and detection
	const VideoProcessingContext videoData(inputFormatCtx.get(), inputDecoderCtx.get(), videoStreamIndex, &watermarkObj, height, width, watermarkInterval, inputFramePtr.get());

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
		checkError(!ffmpegPipe.get(), "Error: Could not open FFmpeg pipe");

		timer::start();
		ArrayXXf inputFrame;
		ArrayXXu8 watermarkedFrame;
		//embed watermark on the video frames
		processFrames(videoData, [&](AVFrame* frame, int& framesCount) { embedWatermarkFrame(videoData, inputFrame, watermarkedFrame, framesCount, frame, ffmpegPipe.get()); });
		timer::end();
		cout << "\nWatermark embedding total execution time: " << executionTime(false, timer::elapsedSeconds()) << "\n";
	}

	//realtime watermarked video detection
	else if (inir.GetBoolean("parameters_video", "watermark_detection", false))
	{
		timer::start();
		ArrayXXf inputFrame;
		//detect watermark on the video frames
		const int framesCount = processFrames(videoData, [&](AVFrame* frame, int& framesCount) { detectFrameWatermark(videoData, inputFrame, framesCount, frame); });
		timer::end();

		cout << "\nWatermark detection total execution time: " << executionTime(false, timer::elapsedSeconds()) << "\n";
		cout << "\nWatermark detection average execution time per frame: " << executionTime(showFps, timer::elapsedSeconds() / framesCount) << "\n";
	}
	return EXIT_SUCCESS;
}

//Main frames loop logic for video watermark embedding and detection
int processFrames(const VideoProcessingContext& data, std::function<void(AVFrame*, int&)> processFrame)
{
	const AVPacketPtr packet(av_packet_alloc(), [](AVPacket* pkt) { av_packet_free(&pkt); });
	const AVFramePtr frame(av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); });
	int framesCount = 0;

	// Read video frames loop
	while (av_read_frame(data.inputFormatCtx, packet.get()) >= 0)
	{
		if (!receivedValidVideoFrame(data.inputDecoderCtx, packet.get(), frame.get(), data.videoStreamIndex))
			continue;
		processFrame(frame.get(), framesCount);
	}
	// Ensure all remaining frames are flushed
	avcodec_send_packet(data.inputDecoderCtx, nullptr);
	while (avcodec_receive_frame(data.inputDecoderCtx, frame.get()) == 0)
	{
		if (frame->format == data.inputDecoderCtx->pix_fmt)
			processFrame(frame.get(), framesCount);
	}
	return framesCount;
}

// Embed watermark in a video frame
void embedWatermarkFrame(const VideoProcessingContext& data, Eigen::ArrayXXf& inputFrame, ArrayXXu8& watermarkedFrame, int& framesCount, AVFrame* frame, FILE* ffmpegPipe)
{
	float watermarkStrength;
	const bool embedWatermark = framesCount % data.watermarkInterval == 0;
	//if there is row padding (for alignment), we must copy the data to a contiguous block!
	if (frame->linesize[0] != data.width)
	{
		if (embedWatermark)
		{
			for (int y = 0; y < data.height; y++)
				memcpy(data.inputFramePtr + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);

			//embed the watermark and receive the watermarked eigen array
			inputFrame = Map<ArrayXXu8>(data.inputFramePtr, data.width, data.height).transpose().cast<float>();
			watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).transpose().cast<uint8_t>();
			//write the watermarked image data
			fwrite(watermarkedFrame.data(), 1, data.width * frame->height, ffmpegPipe);
		}
		else
		{
			//write from frame buffer row-by-row the the valid image data (and not the alignment bytes)
			for (int y = 0; y < data.height; y++)
				fwrite(frame->data[0] + y * frame->linesize[0], 1, data.width, ffmpegPipe);
		}
		//always write UV planes as-is
		for (int y = 0; y < data.height / 2; y++)
			fwrite(frame->data[1] + y * frame->linesize[1], 1, data.width / 2, ffmpegPipe);
		for (int y = 0; y < data.height / 2; y++)
			fwrite(frame->data[2] + y * frame->linesize[2], 1, data.width / 2, ffmpegPipe);
	}
	//else, use original pointer, no need to copy data to intermediate buffer
	else
	{
		if (embedWatermark)
		{
			inputFrame = Map<ArrayXXu8>(frame->data[0], data.width, data.height).transpose().cast<float>();
			watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).transpose().cast<uint8_t>();
		}
		// Write original or modified frame to ffmpeg (pipe)
		fwrite(embedWatermark ? watermarkedFrame.data() : frame->data[0], 1, data.width * frame->height, ffmpegPipe);
		fwrite(frame->data[1], 1, data.width * frame->height / 4, ffmpegPipe);
		fwrite(frame->data[2], 1, data.width * frame->height / 4, ffmpegPipe);
	}
	framesCount++;
}

// Detect the watermark for a video frame
void detectFrameWatermark(const VideoProcessingContext& data, ArrayXXf& inputFrame, int& framesCount, AVFrame* frame)
{
	//detect watermark after X frames
	if (framesCount % data.watermarkInterval == 0)
	{
		//if there is row padding (for alignment), we must copy the data to a contiguous block!
		const bool rowPadding = frame->linesize[0] != data.width;
		if (rowPadding)
		{
			for (int y = 0; y < data.height; y++)
				memcpy(data.inputFramePtr + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);
		}
		//run the detection of the watermark
		inputFrame = Map<ArrayXXu8>(rowPadding ? data.inputFramePtr : frame->data[0], data.width, data.height).transpose().cast<float>();
		float correlation = data.watermarkObj->detectWatermark(inputFrame, MASK_TYPE::ME);
		cout << "Correlation for frame: " << framesCount << ": " << correlation << "\n";
	}
	framesCount++;
}

// find the first video stream index
int findVideoStreamIndex(const AVFormatContext* inputFormatCtx)
{
	for (unsigned int i = 0; i < inputFormatCtx->nb_streams; i++)
		if (inputFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
			return i;
	return -1;
}

//open decoder context for video
AVCodecContext* openDecoderContext(const AVCodecParameters* inputCodecParams)
{
	const AVCodec* inputDecoder = avcodec_find_decoder(inputCodecParams->codec_id);
	AVCodecContext* inputDecoderCtx = avcodec_alloc_context3(inputDecoder);
	avcodec_parameters_to_context(inputDecoderCtx, inputCodecParams);
	//multithreading decode
	inputDecoderCtx->thread_count = 0;
	if (inputDecoder->capabilities & AV_CODEC_CAP_FRAME_THREADS)
		inputDecoderCtx->thread_type = FF_THREAD_FRAME;
	else if (inputDecoder->capabilities & AV_CODEC_CAP_SLICE_THREADS)
		inputDecoderCtx->thread_type = FF_THREAD_SLICE;
	else
		inputDecoderCtx->thread_count = 1; //don't use multithreading
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
	int sendPacketResult = avcodec_send_packet(inputDecoderCtx, packet);
	av_packet_unref(packet);
	if (sendPacketResult != 0 || avcodec_receive_frame(inputDecoderCtx, frame) != 0)
		return false;
	return frame->format == AV_PIX_FMT_YUV420P;
}

//calculate execution time in seconds, or show FPS value
string executionTime(const bool showFps, const double seconds) 
{
	return showFps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds);
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