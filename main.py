import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

x = torch.rand(5, 3)
print(x)

plt.plot([np.random.random() for _ in range(100)])
plt.show()
