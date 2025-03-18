#include "videoprocessingcontext.hpp"
#include "Watermark.hpp"
#include <cstdint>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

VideoProcessingContext::VideoProcessingContext(AVFormatContext* inputCtx, AVCodecContext* decoderCtx, const int streamIdx,
    Watermark* watermark, const int h, const int w, const int interval, uint8_t* pixelData)
    : inputFormatCtx(inputCtx), inputDecoderCtx(decoderCtx), videoStreamIndex(streamIdx), watermarkObj(watermark),
    height(h), width(w), watermarkInterval(interval), inputFramePtr(pixelData)
{ }