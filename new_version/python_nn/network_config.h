#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <stdint.h>

#define NET_INPUTS 10
#define NET_L1 5
#define NET_L2 1

// ===== INPUT QUANTIZATION =====
static const float INPUT_SCALE = 1.0f;
static const int INPUT_ZERO_POINT = 0;

// ===== LAYER 1 =====
static const int8_t L1_W[NET_L1][NET_INPUTS] = {
    { 13, 3, 6, 15, 7, 14, 5, 9, 5, 0 },
    { 15, 0, 13, 10, 7, 8, 13, 4, 15, 3 },
    { 5, 15, 2, 11, 0, 15, 2, 13, 1, 11 },
    { 0, 14, 14, 5, 12, 2, 12, 11, 1, 15 },
    { 3, 6, 0, 2, 15, 5, 1, 4, 5, 3 }
};

static const int32_t L1_B[NET_L1] = { -3, 0, -5, 5, 1
 };

static const float L1_W_SCALE[NET_L1] = { 0.03596557304263115, 0.11411590874195099, 0.09793657809495926, 0.08506684005260468, 0.10748320072889328
 };
static const int L1_W_ZP[NET_L1] = { 11, 9, 8, 9, 5
 };

static const float ACT1_SCALE = 1.1105035146077473f;
static const int ACT1_ZP = 0;

// ===== LAYER 2 =====
static const int8_t L2_W[NET_L2][NET_L1] = {
    { 0, 14, 13, 15, 14 }
};

static const int32_t L2_B[NET_L2] = { -7
 };

static const float L2_W_SCALE[NET_L2] = { 0.3405015468597412
 };
static const int L2_W_ZP[NET_L2] = { 1
 };

#endif
