#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// map of internal values to binary codes
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

void format_binary(uint16_t val, char *out) {
    out[0] = '0'; out[1] = 'b';
    for (int i = 14; i >= 0; i--)
        out[2 + 14 - i] = (val & (1 << i)) ? '1' : '0';
    out[17] = '\0';
}

int8_t code_to_val(Code c) {
    switch (c) {
        case C_NEG_7: return -7;
        case C_NEG_6: return -6;
        case C_NEG_5: return -5;
        case C_NEG_4: return -4;
        case C_NEG_3: return -3;
        case C_NEG_2: return -2;
        case C_NEG_1: return -1;
        case C_POS_1: return  1;
        case C_POS_2: return  2;
        case C_POS_3: return  3;
        case C_POS_4: return  4;
        case C_POS_5: return  5;
        case C_POS_6: return  6;
        case C_POS_7: return  7;
        default:      return  0;
    }
}

// generate 98 new 15‑bit codes
#define WIDE_COUNT 98   
uint16_t binary_neg_codes[49];
uint16_t binary_pos_codes[49];

void generate_binary_codes() {
    bool used[0x8000] = { false };
    // mark original codes as used
    Code orig[] = { C_NEG_7, C_NEG_6, C_NEG_5, C_NEG_4, C_NEG_3, C_NEG_2, C_NEG_1,
                    C_POS_1, C_POS_2, C_POS_3, C_POS_4, C_POS_5, C_POS_6, C_POS_7 };
    for (int i = 0; i < 14; i++) used[orig[i]] = true;

    int neg_idx = 0, pos_idx = 0;
    for (uint16_t candidate = 1; candidate < 0x7FFF; candidate++) {
        if (used[candidate]) continue;
        if (neg_idx < 49) {
            binary_neg_codes[neg_idx++] = candidate;   
            used[candidate] = true;
        } else if (pos_idx < 49) {
            binary_pos_codes[pos_idx++] = candidate;
            used[candidate] = true;
        } else {
            break;
        }
    }
    if (neg_idx != 49 || pos_idx != 49) {
        fprintf(stderr, "Error: not enough unused 15‑bit codes!\n");
    }
}

void write_mult_lut(FILE *f) {
    fprintf(f, "// AUTO-GENERATED MULTIPLICATION LUT (returns binary code)\n");
    fprintf(f, "#ifndef MULT_LUT_H\n#define MULT_LUT_H\n\n");
    fprintf(f, "#include <stdint.h>\n\n");

    fprintf(f, "typedef struct {\n");
    fprintf(f, "    uint16_t a;\n");
    fprintf(f, "    uint16_t b;\n");
    fprintf(f, "    uint16_t result; // binary code from [-49; 49]\n");
    fprintf(f, "} MulEntry;\n\n");

    Code all_codes[] = {
        C_NEG_7, C_NEG_6, C_NEG_5, C_NEG_4, C_NEG_3, C_NEG_2, C_NEG_1,
        C_POS_1, C_POS_2, C_POS_3, C_POS_4, C_POS_5, C_POS_6, C_POS_7
    };

    fprintf(f, "static const MulEntry mult_lut[196] = {\n");
    char bin_a[18], bin_b[18], bin_res[18];
    int count = 0;

    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            int real_prod = code_to_val(all_codes[i]) * code_to_val(all_codes[j]);
            uint16_t wcode;
            if (real_prod < 0)
                wcode = binary_neg_codes[49 + real_prod];
            else
                wcode = binary_pos_codes[real_prod - 1];

            format_binary(all_codes[i], bin_a);
            format_binary(all_codes[j], bin_b);
            format_binary(wcode, bin_res);

            fprintf(f, "    {%s, %s, %s}", bin_a, bin_b, bin_res);
            count++;
            if (count < 196) fprintf(f, ",");
            fprintf(f, "\n");
        }
    }
    fprintf(f, "};\n\n#endif // MULT_LUT_H\n");
}

void write_add_lut(FILE *f) {
    fprintf(f, "// AUTO-GENERATED ADDITION LUT (two binary codes return real sum)\n");
    fprintf(f, "#ifndef ADD_LUT_H\n#define ADD_LUT_H\n\n");
    fprintf(f, "#include <stdint.h>\n\n");

    fprintf(f, "typedef struct {\n");
    fprintf(f, "    uint16_t a;\n");
    fprintf(f, "    uint16_t b;\n");
    fprintf(f, "    int16_t  sum;\n");
    fprintf(f, "} AddEntry;\n\n");

    // gather all binary codes in order
    uint16_t all_binary[98];
    for (int i = 0; i < 49; i++) all_binary[i] = binary_neg_codes[i];       
    for (int i = 0; i < 49; i++) all_binary[49 + i] = binary_pos_codes[i];   


    fprintf(f, "static const AddEntry add_lut[%d] = {\n", 98 * 98);
    char bin_a[18], bin_b[18];
    int count = 0;

    for (int i = 0; i < 98; i++) {
        int val_a = (i < 49) ? -(49 - i) : (i - 49 + 1);
        for (int j = 0; j < 98; j++) {
            int val_b = (j < 49) ? -(49 - j) : (j - 49 + 1);
            int sum_real = val_a + val_b;

            format_binary(all_binary[i], bin_a);
            format_binary(all_binary[j], bin_b);
            fprintf(f, "    {%s, %s, %4d}", bin_a, bin_b, sum_real);
            count++;
            if (count < 98 * 98) fprintf(f, ",");
            fprintf(f, "\n");
        }
    }
    fprintf(f, "};\n\n#endif // ADD_LUT_H\n");
}

void write_add_ops(FILE *f) {
    fprintf(f, "// AUTO-GENERATED\n");
    fprintf(f, "// Helper function to add two binary codes\n");
    fprintf(f, "#ifndef ADD_OPS_H\n#define ADD_OPS_H\n\n");
    fprintf(f, "#include <stdint.h>\n");
    fprintf(f, "#include \"add_lut.h\"\n\n");

    fprintf(f, "static inline int add_binary_codes(uint16_t a, uint16_t b) {\n");
    fprintf(f, "    for (int i = 0; i < %d; i++) {\n", 98 * 98);
    fprintf(f, "        if (add_lut[i].a == a && add_lut[i].b == b)\n");
    fprintf(f, "            return add_lut[i].sum;\n");
    fprintf(f, "    }\n");
    fprintf(f, "    return 0; // should never happen\n");
    fprintf(f, "}\n\n#endif // ADD_OPS_H\n");
}

int main() {
    generate_binary_codes();
    
    FILE *fm = fopen("mult_lut.h", "w");
    if (!fm) { perror("mult_lut.h"); return 1; }
    write_mult_lut(fm);
    fclose(fm);
    puts("Generated mult_lut.h");

    FILE *fa = fopen("add_lut.h", "w");
    if (!fa) { perror("add_lut.h"); return 1; }
    write_add_lut(fa);
    fclose(fa);
    puts("Generated add_lut.h");

    FILE *fo = fopen("add_ops.h", "w");
    if (!fo) { perror("add_ops.h"); return 1; }
    write_add_ops(fo);
    fclose(fo);
    puts("Generated add_ops.h");

    return 0;
}