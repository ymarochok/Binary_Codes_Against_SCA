#!/usr/bin/env python3
import sys
import re
import random


"""
000101100101110
110001110101111
000001100110000
110101110110001
010101110110010
100001100110011
100101110110100
010001100110101
110001100110110
000101110110111
000101110111000
110001100111001
010001100111010
100101110111011
"""
# ------------------------------------------------------------
# Hamming (15,11) systematic encoder
# ------------------------------------------------------------
def hamming_encode(data11):
    """ data11: 11‑bit integer (0..2047) -> 15‑bit codeword """
    d = [(data11 >> i) & 1 for i in range(11)]
    p0 = d[0] ^ d[1] ^ d[3] ^ d[4] ^ d[6] ^ d[8] ^ d[10]
    p1 = d[0] ^ d[2] ^ d[3] ^ d[5] ^ d[6] ^ d[9] ^ d[10]
    p2 = d[1] ^ d[2] ^ d[3] ^ d[7] ^ d[8] ^ d[9] ^ d[10]
    p3 = d[4] ^ d[5] ^ d[6] ^ d[7] ^ d[8] ^ d[9] ^ d[10]
    cw = (p0 << 14) | (p1 << 13) | (d[0] << 12) | (p2 << 11) | \
         (d[1] << 10) | (d[2] << 9) | (d[3] << 8) | (p3 << 7) | \
         (d[4] << 6) | (d[5] << 5) | (d[6] << 4) | (d[7] << 3) | \
         (d[8] << 2) | (d[9] << 1) | d[10]
    return cw

# ------------------------------------------------------------
# Parse the original header file
# ------------------------------------------------------------
def parse_int_array(text, name):
    """ Extract a 2D or 1D integer array from C source. """
    pattern = rf'{name}\s*\[\s*[^\]]*\]\s*=\s*\{{(.*?)\}}'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find {name} in header")
    content = match.group(1)
    # Split by commas, remove whitespace, convert to int
    parts = re.split(r',\s*', content.strip())
    return [int(p) for p in parts if p.strip()]

def parse_weights_2d(text, name, rows, cols):
    flat = parse_int_array(text, name)
    return [flat[i*cols:(i+1)*cols] for i in range(rows)]

# ------------------------------------------------------------
# Main conversion
# ------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python encode_weights.py input_config.h output_config.h")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r') as f:
        content = f.read()

    # Original weights (extracted from header)
    NET_INPUTS = 10
    NET_L1 = 6
    NET_L2 = 1

    L1_W = parse_weights_2d(content, 'L1_W', NET_L1, NET_INPUTS)
    L2_W = parse_weights_2d(content, 'L2_W', NET_L2, NET_L1)
    L1_B = parse_int_array(content, 'L1_B')
    L2_B = parse_int_array(content, 'L2_B')
    L1_M0 = parse_int_array(content, 'L1_M0')
    L1_N = parse_int_array(content, 'L1_N')
    L2_M0 = parse_int_array(content, 'L2_M0')
    L2_N = parse_int_array(content, 'L2_N')
    OUTPUT_SCALE_str = re.search(r'OUTPUT_SCALE\s*=\s*([0-9.eE+-fF]+)', content).group(1)

    # --------------------------------------------------------
    # Build mapping value -> codeword (fixed seed for reproducibility)
    # --------------------------------------------------------
    random.seed(42)
    value_to_cw = []
    cw_to_idx = {}

    for val in range(-7, 8):
        idx = val + 7
        data = random.randint(0, 0x7FF)
        cw = hamming_encode(data)
        value_to_cw.append(cw)
        cw_to_idx[cw] = idx

    # --------------------------------------------------------
    # Create multiplication LUT: mul_lut[idx1][idx2] = val1 * val2
    # --------------------------------------------------------
    mul_lut = [[0]*15 for _ in range(15)]
    for i in range(15):
        vi = i - 7
        for j in range(15):
            vj = j - 7
            mul_lut[i][j] = vi * vj

    # --------------------------------------------------------
    # Convert weight matrices to codewords
    # --------------------------------------------------------
    L1_W_cw = [[value_to_cw[w + 7] for w in row] for row in L1_W]
    L2_W_cw = [[value_to_cw[w + 7] for w in row] for row in L2_W]

    # --------------------------------------------------------
    # Write output header
    # --------------------------------------------------------
    with open(output_file, 'w') as f:
        f.write("#ifndef NETWORK_CONFIG_H\n")
        f.write("#define NETWORK_CONFIG_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NET_INPUTS {NET_INPUTS}\n")
        f.write(f"#define NET_L1 {NET_L1}\n")
        f.write(f"#define NET_L2 {NET_L2}\n\n")

        f.write("// ===== VALUE <-> CODEWORD MAPPING =====\n")
        f.write("static const uint16_t value_to_cw[15] = {\n    ")
        f.write(", ".join(f"0x{cw:04X}" for cw in value_to_cw))
        f.write("\n};\n\n")

        f.write("// Fast lookup: codeword (15 bits) -> index (0..14) or -1\n")
        f.write("static const int16_t cw_to_idx[32768] = {\n")
        arr = ["-1"] * 32768
        for cw, idx in cw_to_idx.items():
            arr[cw] = str(idx)
        for i in range(0, 32768, 16):
            f.write("    " + ", ".join(arr[i:i+16]) + ",\n")
        f.write("};\n\n")

        f.write("// Multiplication LUT (original integer product)\n")
        f.write("static const int8_t mul_lut[15][15] = {\n")
        for row in mul_lut:
            f.write("    {" + ", ".join(f"{v:4d}" for v in row) + "},\n")
        f.write("};\n\n")

        f.write("// ===== LAYER 1 (encoded weights) =====\n")
        f.write("static const uint16_t L1_W_cw[NET_L1][NET_INPUTS] = {\n")
        for row in L1_W_cw:
            f.write("    {" + ", ".join(f"0x{cw:04X}" for cw in row) + "},\n")
        f.write("};\n\n")

        f.write("static const int32_t L1_B[NET_L1] = { " + ", ".join(str(b) for b in L1_B) + " };\n")
        f.write("static const int32_t L1_M0[NET_L1] = { " + ", ".join(str(m) for m in L1_M0) + " };\n")
        f.write("static const int32_t L1_N[NET_L1] = { " + ", ".join(str(n) for n in L1_N) + " };\n\n")

        f.write("// ===== LAYER 2 (encoded weights) =====\n")
        f.write("static const uint16_t L2_W_cw[NET_L2][NET_L1] = {\n")
        for row in L2_W_cw:
            f.write("    {" + ", ".join(f"0x{cw:04X}" for cw in row) + "}\n")
        f.write("};\n\n")

        f.write("static const int32_t L2_B[NET_L2] = { " + ", ".join(str(b) for b in L2_B) + " };\n")
        f.write("static const int32_t L2_M0[NET_L2] = { " + ", ".join(str(m) for m in L2_M0) + " };\n")
        f.write("static const int32_t L2_N[NET_L2] = { " + ", ".join(str(n) for n in L2_N) + " };\n\n")

        f.write(f"static const float OUTPUT_SCALE = {OUTPUT_SCALE_str};\n\n")
        f.write("#endif\n")

    print(f"Generated {output_file} successfully.")

if __name__ == "__main__":
    main()