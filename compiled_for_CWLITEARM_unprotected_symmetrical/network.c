#include "network_config.h"
#include <stdint.h>

/*
110100010000001
010100010000010
100000000000011
100100010000100
010000000000101
110000000000110
000100010000111
000100010001000
110000000001001
010000000001010
100100010001011
100000000001100
010100010001101
110100010001110

*/

/**
 * Procedural Forward Pass
 * Iterates through layers using a loop to reduce code duplication.
 */
int32_t network_forward(const int8_t input[])
{
    // We use two buffers to swap between input and output of layers
    // Layer 1 input is the function argument 'input'
    // Layer 1 output is stored in hidden_q
    int8_t hidden_q[NET_L1];
    
    // Pointer to current input for the active layer loop
    const int8_t* current_input = input;
    int32_t current_input_size = NET_INPUTS;
    
    // We iterate 2 times for 2 layers
    for (int layer = 0; layer < 2; layer++) {

        for(int n=0; n<20; n++) { __asm__ __volatile__ ("nop"); }
        int num_neurons = (layer == 0) ? NET_L1 : 1;
        
        for (int o = 0; o < num_neurons; o++) {
            int32_t acc = 0;
            
            // 1. Dot Product
            for (int i = 0; i < current_input_size; i++) {
                // Select weights based on current layer
                int8_t w = (layer == 0) ? L1_W[o][i] : L2_W[o][i];
                acc += (int32_t)current_input[i] * w;
            }

            // 2. Add Bias and Scale
            int32_t bias = (layer == 0) ? L1_B[o] : L2_B[o];
            int32_t m0   = (layer == 0) ? L1_M0[o] : L2_M0[o];
            int32_t n    = (layer == 0) ? L1_N[o] : L2_N[o];
            
            acc += bias;

            // 3. Fixed-point multiplication with rounding
            int64_t temp = (int64_t)acc * m0;
            temp += (1LL << (31 + n - 1));
            int32_t activation = (int32_t)(temp >> (31 + n));

            // 4. Layer Specific Logic (ReLU for hidden, identity for output)
            if (layer == 0) {
                // Clamp to 3-bit/4-bit range [0, 7]
                if (activation < 0) activation = 0;
                if (activation > 7) activation = 7;
                hidden_q[o] = (int8_t)activation;
            } else {
                // Layer 2 is the final logit, return immediately
                return activation;
            }
        }
        
        // Prepare for the next layer
        current_input = hidden_q;
        current_input_size = NET_L1;
    }

    return 0; // Should not reach here
}