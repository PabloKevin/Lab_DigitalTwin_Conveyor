// facefind.h - find the Rubik cube face (a square) in a BGR image and make the upright model input.
// C++ port of classifier/preprocess.py (same algorithm, same parameters - see that file for the explanation).
// Plain C++ (no Arduino / OpenCV): compiled for the ESP32 and, for the parity test, as a PC shared library
// (classifier/tests/test_facefind_parity.py compares it with the Python version on the dataset).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int downsample;
  int win_x1, win_y1, win_x2, win_y2;      // window in image px; win_x2 <= 0 -> whole image
  float chroma_gain;
  int mode_percentile;                     // 0 fixed lo/hi, 1 percentile
  int lo, hi;
  float hi_percentile, lo_ratio;
  int min_edge_px;
  int close_r, open_r;
  int min_area_px, max_area_px, edge_margin_px;
  float max_side_ratio, min_rect_fill, side_cm_min, side_cm_max;
  float merge_gap_cm, merge_max_deg;
  int merge_min_px;
  float corner_probe_frac;
  int min_corner_hits;
  float px_per_cm;
  int input_size;
  float inset_frac;
} ff_params;

enum { FF_NONE = 0, FF_PARTIAL = 1, FF_OK = 2 };

typedef struct {
  int status;             // FF_NONE / FF_PARTIAL (touches the left/right edge) / FF_OK
  int bbox[4];            // x1, y1, x2, y2 in image px (x2, y2 exclusive)
  float corners[8];       // tl, tr, br, bl (x, y) in image px, shrunk by inset_frac  (FF_OK only)
  float size_cm[2];       // sides of the rotated rectangle (FF_OK only)
  float angle;            // degrees in [0, 90)
  int area_px;            // blob area in image px
  int n_blobs;            // components + merged groups examined
} ff_result;

// Optional profiling: set ff_clock_us (e.g. to esp_timer_get_time) and read ff_timing_us after ff_process:
//   [0] downsample + gradient  [1] non-max + hysteresis  [2] mask (close / fill / open)
//   [3] components + square filter  [4] fragment merging  [5] face crop (warp)
#define FF_N_TIMING 6
extern int64_t (*ff_clock_us)(void);
extern int32_t ff_timing_us[FF_N_TIMING];

// bgr: image, `stride` bytes per row (>= 3 * w). crop_bgr: input_size * input_size * 3 bytes, or NULL.
// Returns status. Memory: allocates ~20 bytes per downsampled pixel while running (std::vector, freed on return).
int ff_process(const uint8_t *bgr, int w, int h, int stride, const ff_params *p, ff_result *r, uint8_t *crop_bgr);

#ifdef __cplusplus
}
#endif
