#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <stdint.h>

#define NET_INPUTS 10
#define NET_L1 5
#define NET_L2 1

// ===== INPUT QUANTIZATION =====
static const float INPUT_SCALE = 1.1428571428571428f;   // for reference only (not used in integer code)
static const int INPUT_ZERO_POINT = 0; // 0 for symmetric

// ===== LAYER 1 =====
static const int8_t L1_W[NET_L1][NET_INPUTS] = {
    { 3, -4, -3, 3, -7, 7, -1, 1, 1, -1 },
    { 2, -1, 3, -2, 1, -5, 7, 0, 3, 2 },
    { -6, 7, -5, 4, 1, 0, -5, 5, -6, 7 },
    { -2, -6, 1, -7, 7, 0, 0, -5, 4, -4 },
    { -3, -7, -7, -3, 1, 2, 4, 4, 3, 6 }
};

static const int32_t L1_B[NET_L1] = { 0, 0, 0, -2, 3 };

// Multiplier (M0) and shift (n) per output channel
static const int32_t L1_M0[NET_L1] = { 1272861529, 1445560652, 1968316487, 1941327310, 1391049214 };
static const int32_t L1_N[NET_L1] = { 4, 4, 5, 5, 4 };

static const int ACT1_SCALE_INV = 0; // optional, if needed

// ===== LAYER 2 =====
static const int8_t L2_W[NET_L2][NET_L1] = {
    { 6, 5, 7, 6, -3 }
};

static const int32_t L2_B[NET_L2] = { -1 };

static const int32_t L2_M0[NET_L2] = { 1078939328255 };
static const int32_t L2_N[NET_L2] = { 0 };

// Output dequantization scale (for converting logit to float probability)
static const float OUTPUT_SCALE = 0.00390625f;

#endif
