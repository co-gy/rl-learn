import os
import matplotlib.pyplot as plt
import numpy as np

root = os.getcwd() + "/result/"
with open(root + "sarsa/length.txt") as f:
    length = list(map(lambda line: float(line.strip()), f.readlines()))
with open(root + "sarsa/reward.txt") as f:
    reward = list(map(lambda line: float(line.strip()), f.readlines()))

plt.subplot(2, 1, 1)
plt.plot(np.arange(len(length)), length, linewidth=0.5)
plt.subplot(2, 1, 2)
plt.plot(np.arange(len(reward)), reward, linewidth=0.5)
plt.show()
