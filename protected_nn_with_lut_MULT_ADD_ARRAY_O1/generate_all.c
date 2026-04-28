#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

// value exchnaging (used only to generate LUTs, during network execution any 
// data which can reveal meaning of binary codes is unavailable

typedef enum {
    C_NEG_7 = 0b110100010000001,
    C_NEG_6 = 0b010100010000010,
    C_NEG_5 = 0b100000000000011,
    C_NEG_4 = 0b100100010000100,
    C_NEG_3 = 0b010000000000101,
    C_NEG_2 = 0b110000000000110,
    C_NEG_1 = 0b000100010000111,
    C_POS_1 = 0b000100010001000,
    C_POS_2 = 0b110000000001001,
    C_POS_3 = 0b010000000001010,
    C_POS_4 = 0b100100010001011,
    C_POS_5 = 0b100000000001100,
    C_POS_6 = 0b010100010001101,
    C_POS_7 = 0b110100010001110
} Code;

// format a 15-bit value as "0bxxxxxxxxxxxxxxx" string
void format_binary(uint16_t val, char *out) {
    out[0] = '0'; out[1] = 'b';
    for (int i = 14; i >= 0; i--)
        out[2 + 14 - i] = (val & (1 << i)) ? '1' : '0';
    out[17] = '\0';
}

// used only to generate LUT
int8_t code_to_val(Code c) {
    switch (c) {
        case C_NEG_7: return -7; case C_NEG_6: return -6; case C_NEG_5: return -5;
        case C_NEG_4: return -4; case C_NEG_3: return -3; case C_NEG_2: return -2;
        case C_NEG_1: return -1; case C_POS_1: return  1; case C_POS_2: return  2;
        case C_POS_3: return  3; case C_POS_4: return  4; case C_POS_5: return  5;
        case C_POS_6: return  6; case C_POS_7: return  7; default:      return  0;
    }
}

// generate 98 new 15‑bit codes
#define WIDE_COUNT 98
uint16_t binary_neg_codes[49];   // index 0 = -49, … 48 = -1
uint16_t binary_pos_codes[49];   // index 0 = +1,  … 48 = +49

void generate_binary_codes() {
    bool used[0x8000] = { false };
    Code orig[] = { C_NEG_7, C_NEG_6, C_NEG_5, C_NEG_4, C_NEG_3, C_NEG_2, C_NEG_1,
                    C_POS_1, C_POS_2, C_POS_3, C_POS_4, C_POS_5, C_POS_6, C_POS_7 };
    for (int i = 0; i < 14; i++) used[orig[i]] = true;

    int neg_idx = 0, pos_idx = 0;
    for (uint16_t c = 1; c < 0x7FFF; c++) {
        if (used[c]) continue;
        if (neg_idx < 49) { binary_neg_codes[neg_idx++] = c; used[c] = true; }
        else if (pos_idx < 49) { binary_pos_codes[pos_idx++] = c; used[c] = true; }
        else break;
    }
}

// LUT files generators

void write_orig_list(FILE *f) {
    Code orig[] = { C_NEG_7, C_NEG_6, C_NEG_5, C_NEG_4, C_NEG_3, C_NEG_2, C_NEG_1,
                    C_POS_1, C_POS_2, C_POS_3, C_POS_4, C_POS_5, C_POS_6, C_POS_7 };
    char bin[18];

    fprintf(f, "#ifndef ORIG_LIST_H\n#define ORIG_LIST_H\n\n");
    fprintf(f, "static const uint16_t orig_code_list[14] = {\n");
    for (int i = 0; i < 14; i++) {
        format_binary((uint16_t)orig[i], bin);
        fprintf(f, "    %s%s\n", bin, i < 13 ? "," : "");
    }
    fprintf(f, "};\n\n#endif\n");
}

void write_binary_list(FILE *f) {
    char bin[18];

    fprintf(f, "// Sorted list of the 98 binary codes (matching LUT index)\n");
    fprintf(f, "#ifndef WIDE_LIST_H\n#define WIDE_LIST_H\n\n");
    fprintf(f, "static const uint16_t binary_code_list[98] = {\n");

    for (int i = 0; i < 49; i++) {
        format_binary(binary_neg_codes[i], bin);
        fprintf(f, "    %s,   // index %2d -> value %d\n", bin, i, -49 + i);
    }

    for (int i = 0; i < 49; i++) {
        format_binary(binary_pos_codes[i], bin);
        fprintf(f, "    %s%s   // index %2d -> value %d\n",
                bin, i < 48 ? "," : " ", 49 + i, i + 1);
    }
    fprintf(f, "};\n\n#endif\n");
}

void write_mult_lut(FILE *f) {
    Code orig[] = { C_NEG_7, C_NEG_6, C_NEG_5, C_NEG_4, C_NEG_3, C_NEG_2, C_NEG_1,
                    C_POS_1, C_POS_2, C_POS_3, C_POS_4, C_POS_5, C_POS_6, C_POS_7 };
    char bin[18];

    fprintf(f, "// Multiplication LUT: 14x14 -> binary code (binary)\n");
    fprintf(f, "#ifndef MULT_LUT_H\n#define MULT_LUT_H\n\n");
    fprintf(f, "static const uint16_t mult_lut[14][14] = {\n");
    for (int i = 0; i < 14; i++) {
        fprintf(f, "    {");
        for (int j = 0; j < 14; j++) {
            int prod = code_to_val(orig[i]) * code_to_val(orig[j]);
            uint16_t wcode = prod < 0 ? binary_neg_codes[49 + prod] : binary_pos_codes[prod - 1];
            format_binary(wcode, bin);
            fprintf(f, "%s", bin);
            if (j < 13) fprintf(f, ", ");
        }
        fprintf(f, "}%s\n", i < 13 ? "," : "");
    }
    fprintf(f, "};\n\n#endif\n");
}

void write_add_lut(FILE *f) {
    fprintf(f, "// Addition LUT: 98x98 returns decimal integers\n");
    fprintf(f, "#ifndef ADD_LUT_H\n#define ADD_LUT_H\n\n");
    fprintf(f, "static const int16_t add_lut[98][98] = {\n");
    for (int i = 0; i < 98; i++) {
        int val_a = i < 49 ? -(49 - i) : (i - 49 + 1);
        fprintf(f, "    {");
        for (int j = 0; j < 98; j++) {
            int val_b = j < 49 ? -(49 - j) : (j - 49 + 1);
            fprintf(f, "%5d", val_a + val_b);
            if (j < 97) fprintf(f, ", ");
        }
        fprintf(f, "}%s\n", i < 97 ? "," : "");
    }
    fprintf(f, "};\n\n#endif\n");
}

void write_code_ops(FILE *f) {
    fprintf(f, "// Fast operations using small index lookup\n");
    fprintf(f, "#ifndef CODE_OPS_H\n#define CODE_OPS_H\n\n");
    fprintf(f, "#include <stdint.h>\n\n");
    fprintf(f, "#include \"orig_list.h\"\n");
    fprintf(f, "#include \"binary_list.h\"\n");
    fprintf(f, "#include \"mult_lut.h\"\n");
    fprintf(f, "#include \"add_lut.h\"\n\n");

    fprintf(f, "static inline int8_t orig_code_to_index(uint16_t code) {\n");
    fprintf(f, "    for (int i = 0; i < 14; i++)\n");
    fprintf(f, "        if (orig_code_list[i] == code) return i;\n");
    fprintf(f, "    return -1;\n");
    fprintf(f, "}\n\n");

    fprintf(f, "static inline int8_t binary_code_to_index(uint16_t code) {\n");
    fprintf(f, "    for (int i = 0; i < 98; i++)\n");
    fprintf(f, "        if (binary_code_list[i] == code) return i;\n");
    fprintf(f, "    return -1;\n");
    fprintf(f, "}\n\n");

    fprintf(f, "static inline uint16_t multiply_original(uint16_t a, uint16_t b) {\n");
    fprintf(f, "    return mult_lut[orig_code_to_index(a)][orig_code_to_index(b)];\n");
    fprintf(f, "}\n\n");

    fprintf(f, "static inline int16_t add_binary_codes(uint16_t a, uint16_t b) {\n");
    fprintf(f, "    return add_lut[binary_code_to_index(a)][binary_code_to_index(b)];\n");
    fprintf(f, "}\n\n");

    fprintf(f, "#endif\n");
}

int main() {
    generate_binary_codes();

    FILE *fo = fopen("orig_list.h", "w");
    write_orig_list(fo); fclose(fo);
    puts("Generated orig_list.h");

    FILE *fw = fopen("binary_list.h", "w");
    write_binary_list(fw); fclose(fw);
    puts("Generated binary_list.h");

    FILE *fm = fopen("mult_lut.h", "w");
    write_mult_lut(fm); fclose(fm);
    puts("Generated mult_lut.h");

    FILE *fa = fopen("add_lut.h", "w");
    write_add_lut(fa); fclose(fa);
    puts("Generated add_lut.h");

    FILE *fc = fopen("code_ops.h", "w");
    write_code_ops(fc); fclose(fc);
    puts("Generated code_ops.h");

    return 0;
}