#include "network_config.h"
#include <stdint.h>

// Forward declaration (if needed)
int32_t network_forward(const int8_t input[]);

/**
 * Pure integer forward pass.
 * Input:  input[10] - int8_t in range [-7, 7] (symmetric 4‑bit)
 * Output: logit_q   - int32_t fixed‑point logit (scale = OUTPUT_SCALE)
 *
 * To get probability:
 *   float prob = 1.0f / (1.0f + expf(-logit_q * OUTPUT_SCALE));
 */
int32_t network_forward(const int8_t input[])
{
    
    
    // optimize function for the loop



    // ------------------------------------------------------------
    // Layer 1: Fully‑connected + ReLU
    // ------------------------------------------------------------
    int8_t hidden_q[NET_L1];   // 4‑bit unsigned after ReLU: [0, 7]

    for (int o = 0; o < NET_L1; o++) {
        // 1. Dot product (int8 * int8 -> int32 accumulator)
        int32_t acc = 0;
        for (int i = 0; i < NET_INPUTS; i++) {
            int32_t x = input[i];          // already quantized, zero‑point = 0
            int32_t w = L1_W[o][i];        // symmetric, zero‑point = 0
            acc += x * w;
        }

        // 2. Add bias (already int32 with correct scale)
        acc += L1_B[o];

        // 3. Apply multiplier: scaled = (acc * M0) >> (31 + N)  with rounding
        int64_t temp = (int64_t)acc * L1_M0[o];
        temp += (1LL << (31 + L1_N[o] - 1));   // rounding half
        int32_t activation = (int32_t)(temp >> (31 + L1_N[o]));

        // 4. ReLU + clamp to 4‑bit unsigned range [0, 15]
        if (activation < 0) activation = 0;
        if (activation > 7) activation = 7;

        hidden_q[o] = (int8_t)activation;
    }

    // ------------------------------------------------------------
    // Layer 2: Fully‑connected (output logit)
    // ------------------------------------------------------------
    int32_t acc = 0;

    for (int i = 0; i < NET_L1; i++) {
        // Hidden activations are unsigned 4‑bit, but stored as int8_t.
        // Weights are symmetric int8 (zero‑point = 0).
        int32_t x = hidden_q[i];           // zero‑point = 0 (already non‑negative)
        int32_t w = L2_W[0][i];            // symmetric
        acc += x * w;
    }

    acc += L2_B[0];

    // Apply multiplier for output layer
    int64_t temp = (int64_t)acc * L2_M0[0];
    temp += (1LL << (31 + L2_N[0] - 1));
    int32_t logit_q = (int32_t)(temp >> (31 + L2_N[0]));

    return logit_q;
}