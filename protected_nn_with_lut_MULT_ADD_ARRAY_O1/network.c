#include "network_config.h"
#include "code_ops.h"            // add_binary_codes(), multiply_original()

int32_t network_forward(const uint16_t input[])
{
    // Computation for the first layer
    int8_t hidden_q[NET_L1];   // after ReLU: [0, 7]

    for (int o = 0; o < NET_L1; o++) {
        // get all products of multiplied binary code values
        uint16_t prod_codes[NET_INPUTS];   // each product is a binary code
        for (int i = 0; i < NET_INPUTS; i++) {
            prod_codes[i] = multiply_original(input[i], L1_W[o][i]);
        }

        // add them in pairs using the addition LUT 
        int32_t acc = 0;   // real accumulator
        for (int i = 0; i < NET_INPUTS; i += 2) {
            uint16_t code_a = prod_codes[i];
            uint16_t code_b = prod_codes[i + 1];
            int pair_sum = add_binary_codes(code_a, code_b);
            acc += pair_sum;

        }
        // add bias
        acc += L1_B[o];

        // apply multiplier by formula scaled = (acc * M0) >> (31 + N)  with rounding
        int64_t temp = (int64_t)acc * L1_M0[o];
        temp += (1LL << (31 + L1_N[o] - 1));   // rounding half
        int32_t activation = (int32_t)(temp >> (31 + L1_N[o]));

        // ReLU + clamp to range [0, 7]
        if (activation < 0) activation = 0;
        if (activation > 7) activation = 7;

        hidden_q[o] = (int8_t)activation;
    }

    // Computation for layer 2
    int32_t acc = 0;

    for (int i = 0; i < NET_L1; i++) {
        int32_t x = hidden_q[i];
        int32_t w = L2_W[0][i];
        acc += x * w;
    }

    acc += L2_B[0];

    // apply multiplier for output layer
    int64_t temp = (int64_t)acc * L2_M0[0];
    temp += (1LL << (31 + L2_N[0] - 1));
    int32_t logit_q = (int32_t)(temp >> (31 + L2_N[0]));

    return logit_q;
}