#include "network_config.h"
#include "network.h"
#include <stdio.h>
float network_forward(const float input[])
{
    float hidden[NET_L1];

    for (int i = 0; i < NET_L1; i++) {
        float acc = 0;
        for (int j = 0; j < NET_INPUTS; j++) {
            acc += (float)L1_W[i][j] * SCALE_L1_W * input[j];
        }
        acc += (float)L1_B[i] * SCALE_L1_B;
        hidden[i] = acc > 0 ? acc : 0;
    }

    float out = 0;
    for (int j = 0; j < NET_L1; j++) {
        out += (float)L2_W[0][j] * SCALE_L2_W * hidden[j];
    }
    out += (float)L2_B[0] * SCALE_L2_B;

    return out;
}
