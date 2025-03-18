#pragma once
#include "Watermark.hpp"
#include <cstdint>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

// Struct to hold common data for video watermarking and detection
// holds pointers and references, does not own any resources
struct VideoProcessingContext
{
    AVFormatContext* inputFormatCtx;
    AVCodecContext* inputDecoderCtx;
    const int videoStreamIndex;
    Watermark* watermarkObj;
    const int height;
    const int width;
    const int watermarkInterval;
    uint8_t* inputFramePtr;

    VideoProcessingContext(AVFormatContext* inputCtx, AVCodecContext* decoderCtx, const int streamIdx,
        Watermark* watermark, const int h, const int w, const int interval, uint8_t* pixelData);
};