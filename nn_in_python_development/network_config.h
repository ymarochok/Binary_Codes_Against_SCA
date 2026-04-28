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
    { 2, -5, 7, -4, 1, 1, -1, 0, 7, -6 },
    { -1, 3, -3, 5, -7, 6, -5, 2, 1, -2 },
    { 1, 0, 6, 2, 0, 7, -1, 6, 3, -1 },
    { 1, -4, 4, -1, -4, 6, -7, 5, 0, -2 },
    { 1, -3, 4, -3, 1, 3, -5, 5, -7, 4 }
};

static const int32_t L1_B[NET_L1] = { -5, -2, 13, 62, -2 };

// Multiplier (M0) and shift (n) per output channel
static const int32_t L1_M0[NET_L1] = { 1310508467, 1336017553, 2083085622, 1267674355, 1146804760 };
static const int32_t L1_N[NET_L1] = { 4, 4, 5, 5, 4 };

static const int ACT1_SCALE_INV = 0; // optional, if needed

// ===== LAYER 2 =====
static const int8_t L2_W[NET_L2][NET_L1] = {
    { 6, 5, -2, -7, 6 }
};

static const int32_t L2_B[NET_L2] = { 2 };

static const int32_t L2_M0[NET_L2] = { 575499186395 };
static const int32_t L2_N[NET_L2] = { 0 };


#endif
