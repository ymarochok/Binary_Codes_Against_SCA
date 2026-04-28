#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <stdint.h>

#define NET_INPUTS 10
#define NET_L1 6
#define NET_L2 1

// input quantization parameters
static const float INPUT_SCALE = 1.0f;   // integers are already quantized
static const int INPUT_ZERO_POINT = 0; // 0 for symmetric (this value is in quantization formula)

// weights for the first Layer
static const uint16_t L1_W[NET_L1][NET_INPUTS] = {
    { 0b110000000000110, 0b100100010001011, 0b110100010000001, 0b100000000001100, 0b110000000000110, 0b000100010000111, 0b110000000001001, 0b110000000001001, 0b110100010000001, 0b100100010001011 },
    { 0b010000000000101, 0b110100010000001, 0b100000000000011, 0b110000000000110, 0b000100010000111, 0b010000000000101, 0b110000000001001, 0b000100010000111, 0b110000000001001, 0b000100010000111 },
    { 0b110000000000110, 0b100000000001100, 0b000100010000111, 0b100000000001100, 0b010000000001010, 0b100100010001011, 0b110100010001110, 0b100100010001011, 0b010000000001010, 0b000100010000111 },
    { 0b010000000001010, 0b010000000001010, 0b110000000000110, 0b010000000001010, 0b000100010001000, 0b110100010000001, 0b100100010000100, 0b110100010000001, 0b100000000000011, 0b100000000000011 },
    { 0b000100010000111, 0b100100010000100, 0b100100010001011, 0b010100010000010, 0b110100010001110, 0b010100010000010, 0b100100010001011, 0b000100010000111, 0b000100010001000, 0b000100010000111 },
    { 0b000100010001000, 0b000100010000111, 0b010000000000101, 0b100100010001011, 0b010000000000101, 0b100100010001011, 0b110000000000110, 0b110000000000110, 0b110100010001110, 0b100000000000011 }
};

// biases for the first layer
static const int32_t L1_B[NET_L1] = { -5, 6, 8, 18, -1, -3 };

// Multipliers and shifts present here in the place of quantization scales, which should produce real values
// But as one of conditions for successful work of encoding scheme, and increased optimization for small hardware
// is full integer arithmetic inside network forward function, we exchanged float point scale values on multipliers (integers)
// and values for bit shift.

// multiplier (M0) and shift (n) per output channel
static const int32_t L1_M0[NET_L1] = { 1259918104, 1794699278, 1863060630, 1135936513, 1237108040, 1439804747 };
static const int32_t L1_N[NET_L1] = { 4, 5, 5, 5, 4, 4 };

static const int ACT1_SCALE_INV = 0; // optional, if needed

// weight for layer 2
static const int8_t L2_W[NET_L2][NET_L1] = {
    { 7, -4, -3, -3, 7, 6 }
};

// biases for layer 2
static const int32_t L2_B[NET_L2] = { -1 };

// multipliers and shifts
static const int32_t L2_M0[NET_L2] = { 1353645644 };
static const int32_t L2_N[NET_L2] = { -8 };



#endif
