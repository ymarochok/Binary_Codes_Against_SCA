#include "network_config.h"
#include <stdint.h>

int32_t network_forward(const int8_t input[])
{
    // computation for layer 1
    int8_t hidden_q[NET_L1];

    for (int o = 0; o < NET_L1; o++) {
        int32_t acc = 0;
        for (int i = 0; i < NET_INPUTS; i++) {
            int32_t x = input[i];          // already quantized, zero‑point = 0
            int32_t w = L1_W[o][i];        // symmetric, zero‑point = 0
            acc += x * w;
        }

        acc += L1_B[o];

        int64_t temp = (int64_t)acc * L1_M0[o];
        temp += (1LL << (31 + L1_N[o] - 1));
        int32_t activation = (int32_t)(temp >> (31 + L1_N[o]));

        if (activation < 0) activation = 0;
        if (activation > 7) activation = 7;

        hidden_q[o] = (int8_t)activation;
    }

    // computation for layer 2
    int32_t acc = 0;

    for (int i = 0; i < NET_L1; i++) {
        int32_t x = hidden_q[i];           // zero‑point = 0
        int32_t w = L2_W[0][i];            // symmetric
        acc += x * w;
    }

    acc += L2_B[0];

    // apply multiplier for output layer
    int64_t temp = (int64_t)acc * L2_M0[0];
    temp += (1LL << (31 + L2_N[0] - 1));
    int32_t logit_q = (int32_t)(temp >> (31 + L2_N[0]));

    return logit_q;
}