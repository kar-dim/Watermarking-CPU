#pragma once
#include "eigen_rgb_array.hpp"
#include <INIReader.h>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/packet.h>
#include <libavutil/frame.h>
#include <libavcodec/codec_par.h>
}

enum IMAGE_TYPE
{
	JPG,
	PNG
};

std::string executionTime(const bool showFps, const double seconds);
void exitProgram(const int exitCode);
void saveWatermarkedImage(const std::string& imagePath, const std::string& suffix, const EigenArrayRGB& watermark, const IMAGE_TYPE type);
int testForImage(const INIReader& inir, const int p, const float psnr);
int testForVideo(const std::string& videoFile, const INIReader& inir, const int p, const float psnr);
int findVideoStreamIndex(const AVFormatContext* inputFormatCtx);
AVCodecContext* openDecoderContext(const AVCodecParameters* params);
bool receivedValidVideoFrame(AVCodecContext* inputDecoderCtx, AVPacket* packet, AVFrame* frame, const int videoStreamIndex);