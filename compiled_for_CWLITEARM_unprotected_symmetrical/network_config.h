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
    { -2, 4, -7, 5, -2, -1, 2, 2, -7, 4 },
    { -3, -7, -5, -2, -1, -3, 2, -1, 2, -1 },
    { -2, 5, -1, 5, 3, 4, 7, 4, 3, -1 },
    { 3, 3, -2, 3, 1, -7, -4, -7, -5, -5 },
    { -1, -4, 4, -6, 7, -6, 4, -1, 1, -1 },
    { 1, -1, -3, 4, -3, 4, -2, -2, 7, -5 }
};

static const int32_t L1_B[NET_L1] = { -5, 6, 8, 18, -1, -3 };

// Multiplier (M0) and shift (n) per output channel
static const int32_t L1_M0[NET_L1] = { 1259918104, 1794699278, 1863060630, 1135936513, 1237108040, 1439804747 };
static const int32_t L1_N[NET_L1] = { 4, 5, 5, 5, 4, 4 };

static const int ACT1_SCALE_INV = 0; // optional, if needed

// ===== LAYER 2 =====
static const int8_t L2_W[NET_L2][NET_L1] = {
    { 7, -4, -3, -3, 7, 6 }
};

static const int32_t L2_B[NET_L2] = { -1 };

static const int32_t L2_M0[NET_L2] = { 1353645644 };
static const int32_t L2_N[NET_L2] = { -8 };


#endif
