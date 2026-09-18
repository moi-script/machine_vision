/*
 * AeroSense court/face camera: Freenove ESP32-S3-WROOM CAM + OV5640 as a USB
 * UVC (MJPEG) webcam. The role (left/right/back/face) only changes the USB
 * serial string and frame size, both set by sdkconfig.role.* at build time,
 * so the Pi can tell identical boards apart through udev.
 *
 * Plug into the Pi through the board's NATIVE USB port, not the UART one.
 */
#include <string.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_camera.h"
#include "usb_device_uvc.h"

static const char *TAG = "aero_uvc";

/* Freenove ESP32-S3-WROOM CAM == ESP32-S3-EYE camera pinout */
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D7     16
#define CAM_PIN_D6     17
#define CAM_PIN_D5     18
#define CAM_PIN_D4     12
#define CAM_PIN_D3     10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0     11
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK   13

#if CONFIG_UVC_CAM1_FRAMESIZE_WIDTH == 800
#define AERO_FRAMESIZE FRAMESIZE_SVGA
#define UVC_BUF_SIZE   (160 * 1024)
#else
#define AERO_FRAMESIZE FRAMESIZE_VGA
#define UVC_BUF_SIZE   (100 * 1024)
#endif

static uvc_fb_t s_fb;
static camera_fb_t *s_cam_fb;

static esp_err_t camera_init(void)
{
    camera_config_t cfg = {
        .pin_pwdn = CAM_PIN_PWDN, .pin_reset = CAM_PIN_RESET,
        .pin_xclk = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD, .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7, .pin_d6 = CAM_PIN_D6, .pin_d5 = CAM_PIN_D5,
        .pin_d4 = CAM_PIN_D4, .pin_d3 = CAM_PIN_D3, .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1, .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC, .pin_href = CAM_PIN_HREF,
        .pin_pclk = CAM_PIN_PCLK,
        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0, .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = AERO_FRAMESIZE,
        .jpeg_quality = 14,          /* lower = better; 14 keeps VGA under full-speed USB */
        .fb_count = 2,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_LATEST,
    };
    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed: %s", esp_err_to_name(err));
        return err;
    }
    sensor_t *s = esp_camera_sensor_get();
    ESP_LOGI(TAG, "sensor PID 0x%04x, %dx%d", s->id.PID,
             CONFIG_UVC_CAM1_FRAMESIZE_WIDTH, CONFIG_UVC_CAM1_FRAMESIZE_HEIGT);
    return ESP_OK;
}

static esp_err_t on_start(uvc_format_t format, int width, int height, int rate, void *ctx)
{
    ESP_LOGI(TAG, "host opened stream %dx%d @ %d fps", width, height, rate);
    return ESP_OK;   /* camera already runs in the single mode we advertise */
}

static void on_stop(void *ctx)
{
    ESP_LOGI(TAG, "host closed stream");
}

static uvc_fb_t *on_fb_get(void *ctx)
{
    s_cam_fb = esp_camera_fb_get();
    if (!s_cam_fb) {
        return NULL;
    }
    if (s_cam_fb->len > UVC_BUF_SIZE) {      /* oversize frame: drop, never overflow */
        esp_camera_fb_return(s_cam_fb);
        s_cam_fb = NULL;
        return NULL;
    }
    s_fb.buf = s_cam_fb->buf;
    s_fb.len = s_cam_fb->len;
    s_fb.width = s_cam_fb->width;
    s_fb.height = s_cam_fb->height;
    s_fb.format = UVC_FORMAT_JPEG;
    s_fb.timestamp = s_cam_fb->timestamp;
    return &s_fb;
}

static void on_fb_return(uvc_fb_t *fb, void *ctx)
{
    if (s_cam_fb) {
        esp_camera_fb_return(s_cam_fb);
        s_cam_fb = NULL;
    }
}

void app_main(void)
{
    ESP_ERROR_CHECK(camera_init());

    uint8_t *buf = heap_caps_malloc(UVC_BUF_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    assert(buf);
    uvc_device_config_t uvc = {
        .uvc_buffer = buf,
        .uvc_buffer_size = UVC_BUF_SIZE,
        .start_cb = on_start,
        .fb_get_cb = on_fb_get,
        .fb_return_cb = on_fb_return,
        .stop_cb = on_stop,
        .cb_ctx = NULL,
    };
    ESP_ERROR_CHECK(uvc_device_config(0, &uvc));
    ESP_ERROR_CHECK(uvc_device_init());
    ESP_LOGI(TAG, "UVC device up, serial %s", CONFIG_TUSB_SERIAL_NUM);
}
