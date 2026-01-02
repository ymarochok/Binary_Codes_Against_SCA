"""
Docstring for protecting_nn_with_binary_codes_from_sca.network_implementation_in_python.save_params

This files contains function to carefully save data in txt file.

"""

import numpy as np

def save_txt(filename, data):
    with open(filename, "w") as f:
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1: 
                for v in data:
                    f.write(str(int(v)) + "\n")
            else: 
                for row in data:
                    f.write(" ".join(str(int(x)) for x in row) + "\n")
        else:
            f.write(str(data))

