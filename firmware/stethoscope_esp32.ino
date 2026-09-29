// Digital stethoscope - INMP441 I2S mic -> 16 kHz int16 audio over USB serial.
// Digital High-Pass Filter: 4th-order Butterworth (Cutoff = 100 Hz)
// ESP32 core 2.0.x / 3.x. Board: "ESP32 Dev Module".
#include <driver/i2s.h>
#include <math.h>

#define I2S_SCK      26
#define I2S_WS       25
#define I2S_SD       33
#define I2S_PORT     I2S_NUM_0
#define SAMPLE_RATE  16000
#define BUF_SAMPLES  256

// ---- 100 Hz High-Pass Filter Parameters ----
// Blocks frequencies < 100 Hz (DC offset, motion artifacts, low heart sounds S1/S2)
// Passes frequencies >= 100 Hz (Crackles 100-250Hz, Wheezes 400-1000Hz, Vesicular breath sounds)
#define HPF_CUTOFF_HZ   100.0f
#define HPF_Q1          0.541196f  // Butterworth 4th-order cascade Q factor 1
#define HPF_Q2          1.306563f  // Butterworth 4th-order cascade Q factor 2

const uint8_t SYNC[4] = {0xA5, 0x5A, 0xA5, 0x5A};
int32_t raw[BUF_SAMPLES];
int16_t out[BUF_SAMPLES];

typedef struct {
  float b0, b1, b2, a1, a2;
  float x1, x2, y1, y2;
} Biquad;

Biquad hpf1, hpf2;

void biquadSetHighpass(Biquad *bq, float fc, float fs, float Q) {
  float w0    = 2.0f * (float)M_PI * fc / fs;
  float cosw0 = cosf(w0);
  float sinw0 = sinf(w0);
  float alpha = sinw0 / (2.0f * Q);

  float b0 =  (1.0f + cosw0) / 2.0f;
  float b1 = -(1.0f + cosw0);
  float b2 =  (1.0f + cosw0) / 2.0f;
  float a0 =  1.0f + alpha;
  float a1 = -2.0f * cosw0;
  float a2 =  1.0f - alpha;

  bq->b0 = b0 / a0;
  bq->b1 = b1 / a0;
  bq->b2 = b2 / a0;
  bq->a1 = a1 / a0;
  bq->a2 = a2 / a0;
  bq->x1 = bq->x2 = bq->y1 = bq->y2 = 0.0f;
}

static inline float biquadProcess(Biquad *bq, float x0) {
  float y0 = bq->b0 * x0 + bq->b1 * bq->x1 + bq->b2 * bq->x2
                          - bq->a1 * bq->y1 - bq->a2 * bq->y2;
  bq->x2 = bq->x1; bq->x1 = x0;
  bq->y2 = bq->y1; bq->y1 = y0;
  return y0;
}

void i2sInit() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  i2s_pin_config_t pins = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };
  i2s_set_pin(I2S_PORT, &pins);
  i2s_zero_dma_buffer(I2S_PORT);
}

void setup() {
  Serial.begin(921600);
  delay(300);
  Serial.println("ESP32_STETHO_READY");
  i2sInit();

  // Initialize 4th-order (24 dB/octave) Butterworth 200 Hz High-Pass Filter
  biquadSetHighpass(&hpf1, HPF_CUTOFF_HZ, (float)SAMPLE_RATE, HPF_Q1);
  biquadSetHighpass(&hpf2, HPF_CUTOFF_HZ, (float)SAMPLE_RATE, HPF_Q2);
}

void loop() {
  size_t bytesRead = 0;
  i2s_read(I2S_PORT, raw, sizeof(raw), &bytesRead, portMAX_DELAY);
  int n = bytesRead / 4;

  for (int i = 0; i < n; i++) {
    int32_t v32 = raw[i] >> 14;
    float x = (float)v32;

    // Apply 4th-order Butterworth High-Pass Filter (Cutoff = 200 Hz)
    float f1 = biquadProcess(&hpf1, x);
    float f2 = biquadProcess(&hpf2, f1);

    int32_t v = (int32_t)lroundf(f2);
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    out[i] = (int16_t)v;
  }

  Serial.write(SYNC, 4);
  uint16_t cnt = (uint16_t)n;
  Serial.write((uint8_t*)&cnt, 2);
  Serial.write((uint8_t*)out, n * 2);
}
