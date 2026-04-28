#include "code_multiply.h"
#include "mult_lut.h"

int8_t multiply_codes(uint16_t a, uint16_t b) {
    // linear search through 196 entries
    for(int i = 0; i < 196; i++) {
        if(mult_lut[i].a == a && mult_lut[i].b == b) {
            return mult_lut[i].result;
        }
    }
    return 0;  // invalid code pair
}