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
    { -1, 5, -6, 6, -7, 7, -7, 6, -5, 3 },
    { 1, -3, 1, 2, -4, 7, -7, 5, 1, -2 },
    { 1, -3, -1, 0, -5, 3, -7, 1, -2, -4 },
    { 1, -3, 1, 3, -5, 4, -2, -2, 7, -5 },
    { 3, -7, 7, -5, 2, 2, -6, 6, -2, 0 }
};

static const int32_t L1_B[NET_L1] = { -8, 52, 17, -6, -10 };

// Multiplier (M0) and shift (n) per output channel
static const int32_t L1_M0[NET_L1] = { 2021112782, 1700080132, 2046292996, 1510965346, 1226009491 };
static const int32_t L1_N[NET_L1] = { 5, 5, 5, 4, 4 };

static const int ACT1_SCALE_INV = 0; // optional, if needed

// ===== LAYER 2 =====
static const int8_t L2_W[NET_L2][NET_L1] = {
    { 7, -6, -3, 6, 7 }
};

static const int32_t L2_B[NET_L2] = { 2 };

static const int32_t L2_M0[NET_L2] = { 386098300507 };
static const int32_t L2_N[NET_L2] = { 0 };


#endif
