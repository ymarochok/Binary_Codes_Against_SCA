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
    { 0, -1, 1, -1, 3, 4, 2, 4, 7, 6 },
    { -1, 1, 0, -4, 6, -6, 7, -6, 3, 1 },
    { 3, -7, 6, -4, -1, 4, -6, 4, 2, -2 },
    { -1, -6, -6, -7, -5, 0, -2, -2, -3, 1 },
    { -2, 5, -7, 6, -6, 4, -1, 3, -5, 4 }
};

static const int32_t L1_B[NET_L1] = { 14, -6, -7, 15, -7 };

// Multiplier (M0) and shift (n) per output channel
static const int32_t L1_M0[NET_L1] = { 1792537580, 1379875256, 1348636477, 1249340340, 1267239652 };
static const int32_t L1_N[NET_L1] = { 5, 4, 4, 5, 4 };

static const int ACT1_SCALE_INV = 0; // optional, if needed

// ===== LAYER 2 =====
static const int8_t L2_W[NET_L2][NET_L1] = {
    { -2, 7, 7, -2, 7 }
};

static const int32_t L2_B[NET_L2] = { -5 };

static const int32_t L2_M0[NET_L2] = { 241953716501 };
static const int32_t L2_N[NET_L2] = { 0 };


#endif
