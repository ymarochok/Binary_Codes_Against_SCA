import numpy as np

def save_txt(filename, data):
    with open(filename, "w") as f:
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1:  # vector
                for v in data:
                    f.write(str(int(v)) + "\n")
            else:  # matrix
                for row in data:
                    f.write(" ".join(str(int(x)) for x in row) + "\n")
        else:
            f.write(str(data))

