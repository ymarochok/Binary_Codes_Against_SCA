#ifndef NETWORK_H
#define NETWORK_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t network_forward(const uint16_t input[]);

#ifdef __cplusplus
}
#endif
#endif