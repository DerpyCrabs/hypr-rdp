/* Opaque libavcodec bridge: Rust never depends on FFmpeg struct layouts. */
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>

typedef struct {
    AVCodecContext *codec;
    AVFrame *frame;
    AVPacket *packet;
    struct SwsContext *scale;
    int64_t next_pts;
} HyprNvenc;

void hypr_nvenc_free(HyprNvenc *enc) {
    if (!enc) return;
    sws_freeContext(enc->scale);
    av_packet_free(&enc->packet);
    av_frame_free(&enc->frame);
    avcodec_free_context(&enc->codec);
    free(enc);
}

static void describe_error(char *error, size_t size, const char *operation, int code) {
    char detail[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(code, detail, sizeof(detail));
    snprintf(error, size, "%s: %s", operation, detail);
}

HyprNvenc *hypr_nvenc_new(int width, int height, int64_t bitrate, int fps,
                         int quality, int cqp, char *error, size_t error_size) {
    const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
    if (!codec) {
        snprintf(error, error_size, "FFmpeg was built without h264_nvenc");
        return NULL;
    }
    HyprNvenc *enc = calloc(1, sizeof(*enc));
    if (!enc) {
        describe_error(error, error_size, "allocate encoder", AVERROR(ENOMEM));
        return NULL;
    }
    int result = AVERROR(ENOMEM);
    const char *operation = "allocate FFmpeg objects";
    enc->codec = avcodec_alloc_context3(codec);
    enc->frame = av_frame_alloc();
    enc->packet = av_packet_alloc();
    if (!enc->codec || !enc->frame || !enc->packet) goto fail;
    AVCodecContext *ctx = enc->codec;
    ctx->width = width;
    ctx->height = height;
    ctx->pix_fmt = AV_PIX_FMT_NV12;
    ctx->time_base = (AVRational){1, fps};
    ctx->framerate = (AVRational){fps, 1};
    ctx->gop_size = fps * 2;
    ctx->max_b_frames = 0;
    ctx->bit_rate = cqp ? 0 : bitrate;
    ctx->color_range = AVCOL_RANGE_JPEG;
    ctx->colorspace = AVCOL_SPC_BT709;
    ctx->color_primaries = AVCOL_PRI_BT709;
    ctx->color_trc = AVCOL_TRC_BT709;

#define SET_OPTION(name, value) do { \
    operation = "set NVENC option " name; \
    result = av_opt_set(ctx->priv_data, name, value, 0); \
    if (result < 0) goto fail; \
} while (0)
    SET_OPTION("preset", "p4");
    SET_OPTION("tune", "ull");
    SET_OPTION("zerolatency", "1");
    SET_OPTION("delay", "0");
    SET_OPTION("rc-lookahead", "0");
    SET_OPTION("forced-idr", "1");
    SET_OPTION("profile", "high");
    SET_OPTION("rc", cqp ? "constqp" : "vbr");
    operation = "set NVENC quality";
    result = av_opt_set_int(ctx->priv_data, cqp ? "qp" : "cq", quality, 0);
    if (result < 0) goto fail;
#undef SET_OPTION
    operation = "open h264_nvenc (check NVIDIA driver and GPU)";
    result = avcodec_open2(ctx, codec, NULL);
    if (result < 0) goto fail;
    enc->frame->format = ctx->pix_fmt;
    enc->frame->width = width;
    enc->frame->height = height;
    enc->frame->color_range = ctx->color_range;
    enc->frame->colorspace = ctx->colorspace;
    enc->frame->color_primaries = ctx->color_primaries;
    enc->frame->color_trc = ctx->color_trc;
    operation = "allocate NV12 frame";
    result = av_frame_get_buffer(enc->frame, 32);
    if (result < 0) goto fail;
    enc->scale = sws_getContext(width, height, AV_PIX_FMT_BGRA,
                               width, height, AV_PIX_FMT_NV12,
                               SWS_BILINEAR, NULL, NULL, NULL);
    operation = "create BGRA to NV12 converter";
    result = AVERROR(ENOMEM);
    if (!enc->scale) goto fail;
    operation = "set BT.709 full-range conversion";
    const int *matrix = sws_getCoefficients(SWS_CS_ITU709);
    result = sws_setColorspaceDetails(enc->scale, matrix, 1, matrix, 1, 0, 1 << 16, 1 << 16);
    if (result < 0) goto fail;
    return enc;
fail:
    describe_error(error, error_size, operation, result);
    hypr_nvenc_free(enc);
    return NULL;
}

/* data remains valid until the next encode/free; the caller copies it immediately.
 * Low-latency NVENC must produce exactly the submitted frame, without reordering.
 * A delayed packet is an error rather than silently dropping a damaged region. */
int hypr_nvenc_encode(HyprNvenc *enc, const uint8_t *bgra, int stride, int force_idr,
                      const uint8_t **data, int *size, char *error, size_t error_size) {
    const char *operation = "make NV12 frame writable";
    int result = av_frame_make_writable(enc->frame);
    if (result < 0) goto fail;
    const uint8_t *planes[4] = {bgra, NULL, NULL, NULL};
    int strides[4] = {stride, 0, 0, 0};
    operation = "convert BGRA to NV12";
    result = sws_scale(enc->scale, planes, strides, 0, enc->codec->height,
                       enc->frame->data, enc->frame->linesize);
    if (result != enc->codec->height) { result = AVERROR(EINVAL); goto fail; }
    int64_t pts = enc->next_pts++;
    enc->frame->pts = pts;
    enc->frame->pict_type = force_idr ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
    av_packet_unref(enc->packet);
    operation = "submit NVENC frame";
    result = avcodec_send_frame(enc->codec, enc->frame);
    if (result < 0) goto fail;
    operation = "receive immediate NVENC packet";
    result = avcodec_receive_packet(enc->codec, enc->packet);
    if (result < 0) goto fail;
    if (enc->packet->pts != pts || enc->packet->size <= 0) {
        operation = "NVENC returned a delayed or empty packet";
        result = AVERROR(EINVAL);
        goto fail;
    }
    *data = enc->packet->data;
    *size = enc->packet->size;
    return 0;
fail:
    describe_error(error, error_size, operation, result);
    return result;
}
