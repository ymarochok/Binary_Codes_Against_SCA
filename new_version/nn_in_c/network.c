#include "network_config.h"
#include "network.h"
#include <math.h>

float network_forward(const int8_t input[])
{
    int8_t hidden_q[NET_L1];

    // ===== FC1 =====
    for (int i = 0; i < NET_L1; i++) {

        int32_t acc = 0;

        for (int j = 0; j < NET_INPUTS; j++) {

            // input already quantized
            int32_t x = input[j];
            int32_t w = L1_W[i][j] - L1_W_ZP[i];

            acc += x * w;
        }

        acc += L1_B[i];

        // dequantize to float
        float out = acc * L1_W_SCALE[i];

        // ReLU
        if (out < 0) out = 0;

        // requantize activation
        int q = (int)roundf(out / ACT1_SCALE) + ACT1_ZP;

        if (q < 0) q = 0;
        if (q > 15) q = 15;

        hidden_q[i] = (int8_t)q;
    }

    // ===== FC2 =====
    int32_t acc = 0;

    for (int j = 0; j < NET_L1; j++) {

        int32_t x = hidden_q[j] - ACT1_ZP;
        int32_t w = L2_W[0][j] - L2_W_ZP[0];

        acc += x * w;
    }

    acc += L2_B[0];

    float out = acc * (ACT1_SCALE * L2_W_SCALE[0]);

    return out;
}


#include "network_config.h"
#include "network.h"
#include <math.h>

float network_forward(const int8_t input[])
{
    int8_t hidden_q[NET_L1];

    // ===== FC1 =====
    for (int i = 0; i < NET_L1; i++) {

        int32_t acc = 0;

        for (int j = 0; j < NET_INPUTS; j++) {

            // input already quantized
            int32_t x = input[j];
            int32_t w = L1_W[i][j];
 
            acc += x * w; inputs from -7 to 7 weights are from -7 7
        }

        acc += L1_B[i];

        // dequantize to float
        float out = L1_W_SCALE[i] * (acc  - ZP);

        // ReLU
        if (out < 0) out = 0;

        // requantize activation
        int q = (int)roundf(out / ACT1_SCALE);

        if (q < 0) q = 0;
        if (q > 15) q = 15;

        hidden_q[i] = (int8_t)q;
    }

    // ===== FC2 =====
    int32_t acc = 0;

    for (int j = 0; j < NET_L1; j++) {

        int32_t x = hidden_q[j] - ACT1_ZP;
        int32_t w = L2_W[0][j] - L2_W_ZP[0];

        acc += x * w;
    }

    acc += L2_B[0];

    float out = acc * (ACT1_SCALE * L2_W_SCALE[0]);

    return out;
}