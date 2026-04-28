// Fast operations using small index lookup
#ifndef CODE_OPS_H
#define CODE_OPS_H

#include <stdint.h>

#include "orig_list.h"
#include "binary_list.h"
#include "mult_lut.h"
#include "add_lut.h"

static inline int8_t orig_code_to_index(uint16_t code) {
    for (int i = 0; i < 14; i++)
        if (orig_code_list[i] == code) return i;
    return -1;
}

static inline int8_t binary_code_to_index(uint16_t code) {
    for (int i = 0; i < 98; i++)
        if (binary_code_list[i] == code) return i;
    return -1;
}

static inline uint16_t multiply_original(uint16_t a, uint16_t b) {
    return mult_lut[orig_code_to_index(a)][orig_code_to_index(b)];
}

static inline int16_t add_binary_codes(uint16_t a, uint16_t b) {
    return add_lut[binary_code_to_index(a)][binary_code_to_index(b)];
}

#endif
