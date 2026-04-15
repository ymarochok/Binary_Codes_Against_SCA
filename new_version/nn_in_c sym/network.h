#ifndef NETWORK_H
#define NETWORK_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Pure integer inference: returns fixed-point logit (scale = OUTPUT_SCALE)
int32_t network_forward(const int8_t input[]);

// // Optional wrapper that returns float probability
// float network_forward(const int8_t input[]);

#ifdef __cplusplus
}
#endif
#endif