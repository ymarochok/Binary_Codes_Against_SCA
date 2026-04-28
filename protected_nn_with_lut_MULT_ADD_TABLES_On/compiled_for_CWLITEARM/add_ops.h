// AUTO-GENERATED
// Helper function to add two binary codes
#ifndef ADD_OPS_H
#define ADD_OPS_H

#include <stdint.h>
#include "add_lut.h"

static inline int add_binary_codes(uint16_t a, uint16_t b) {
    for (int i = 0; i < 9604; i++) {
        if (add_lut[i].a == a && add_lut[i].b == b)
            return add_lut[i].sum;
    }
    return 0; // should never happen
}

#endif // ADD_OPS_H
