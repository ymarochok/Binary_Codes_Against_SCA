#include <stdint.h>

#define NET_INPUTS 10
#define NET_L1 5
#define NET_L2 1

// ===== Quantized Scales =====
static const float SCALE_L1_W = 0.43469067982279214f;
static const float SCALE_L1_B = 0.027382299304151342f;

static const float SCALE_L2_W = 0.8496161869595458f;
static const float SCALE_L2_B = 0.21371381623418653f;

// ===== Layer 0 (10 → 5) =====


// weights [5][10]
static const int8_t L1_W[NET_L1][NET_INPUTS] = {
    {  2,  2, -3, -4, -2, -1, -3, -2, -1, -6 },
    { -5,  1, -3,  2, -2, -2, -5, -2, -2,  6 },
    {  3,  2,  3, -2,  4, -2,  2, -3, -2,  1 },
    {  2, -4,  2,  5, -2, -2, -2, -1, -2,  0 },
    { -4,  0, -3, -5, -2, -1,  7, -2,  0,  2 }
};

// bias [5]
static const int8_t L1_B[NET_L1] = { -1, -4,  5, -1,  7 };

// ===== Layer 1 (5 → 1) =====

// weights [1][5]
static const int8_t L2_W[NET_L2][NET_L1] = {
    { 6, 5, 7, 6, 5 }
};

// bias [1]
static const int8_t L2_B[NET_L2] = { -7 };
