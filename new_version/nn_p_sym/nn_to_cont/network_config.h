#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <stdint.h>

#define NET_INPUTS 10
#define NET_L1 6
#define NET_L2 1

// ===== INPUT QUANTIZATION =====
static const float INPUT_SCALE = 1.0f;   // for reference only (not used in integer code)
static const int INPUT_ZERO_POINT = 0; // 0 for symmetric

// ===== LAYER 1 =====
static const int8_t L1_W[NET_L1][NET_INPUTS] = {
    { 1, -5, -1, -1, -3, -4, -6, -7, -5, -3 },
    { -1, 3, -2, -3, 5, -7, 4, -3, -1, 1 },
    { -1, 5, -7, 5, -5, 4, -4, 1, 1, -1 },
    { 1, -2, 2, -3, 1, 2, -4, 3, -7, 4 },
    { 1, -4, 6, -4, 1, 1, -3, -1, 7, -6 },
    { 3, -7, 1, 7, -6, -1, 5, -6, -1, 1 }
};

static const int32_t L1_B[NET_L1] = { 19, -12, -14, -11, -13, -14 };

// Multiplier (M0) and shift (n) per output channel
static const int32_t L1_M0[NET_L1] = { 1891320555, 1395975740, 1179142213, 1618603681, 1149714227, 1953088547 };
static const int32_t L1_N[NET_L1] = { 5, 4, 4, 4, 4, 5 };

static const int ACT1_SCALE_INV = 0; // optional, if needed

// ===== LAYER 2 =====
static const int8_t L2_W[NET_L2][NET_L1] = {
    { -3, 6, 7, 5, 6, 6 }
};

static const int32_t L2_B[NET_L2] = { -6 };

static const int32_t L2_M0[NET_L2] = { 1924724802 };
static const int32_t L2_N[NET_L2] = { -8 };

// Output dequantization scale (for converting logit to float probability)
static const float OUTPUT_SCALE = 0.00390625f;

#endif
