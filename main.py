import os
import numpy as np

C = np.logspace(4, 8, 4)
max_iters = [100, 1500,250, 500]

for c in C:
  for iter in max_iters:
        print(f"logging experiment for C:{c}, iter: {iter}")
        os.system(f"python src/train.py -C_ {c} -iter {iter}")
        # os.system(f"python src/train.py -C {C_} -iter {iter}")