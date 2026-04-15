#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <stdint.h>

#define NET_INPUTS 10
#define NET_L1 5
#define NET_L2 1

// ===== INPUT QUANTIZATION =====
static const float INPUT_SCALE = 1.0f;   // for reference only (not used in integer code)
static const int INPUT_ZERO_POINT = 0; // 0 for symmetric

// ===== LAYER 1 =====
static const int8_t L1_W[NET_L1][NET_INPUTS] = {
    { 1, -6, 7, -7, 4, -2, 0, 1, -1, 1 },
    { -2, -4, -5, -5, -7, -2, -4, 0, 0, 0 },
    { -2, 5, -6, 2, -1, 0, 1, 3, -7, 6 },
    { -1, 1, -3, 3, -4, 4, -2, 0, 7, -5 },
    { -3, 2, -2, 2, 1, 7, 5, 6, 4, 3 }
};

static const int32_t L1_B[NET_L1] = { -3, 16, -5, -4, 19 };

// Multiplier (M0) and shift (n) per output channel
static const int32_t L1_M0[NET_L1] = { 1331150097, 2140821279, 1326745243, 1625594765, 1630292597 };
static const int32_t L1_N[NET_L1] = { 4, 5, 4, 4, 5 };

static const int ACT1_SCALE_INV = 0; // optional, if needed

// ===== LAYER 2 =====
static const int8_t L2_W[NET_L2][NET_L1] = {
    { 7, -2, 7, 7, -3 }
};

static const int32_t L2_B[NET_L2] = { -4 };

static const int32_t L2_M0[NET_L2] = { 1166033511 };
static const int32_t L2_N[NET_L2] = { -8 };

// Output dequantization scale (for converting logit to float probability)
static const float OUTPUT_SCALE = 0.00390625f;

#endif
