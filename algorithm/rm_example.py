import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3.5, 10000)
y = lambda x_: np.tanh(np.array(x_)-1)
max_iteration = 100
make_noise = False

init_guess = 3

w_k = [init_guess]
noise = np.random.uniform(-1, 1, max_iteration)
noise = np.zeros_like(noise)
for k in range(1, max_iteration):
    w_k.append(w_k[-1] - (1/k)*(y(w_k[-1]) + noise[k]))
    plt.plot([w_k[-2], w_k[-2]], [0, y(w_k[-2])], color="black", linewidth=0.5, linestyle="--")
    plt.plot([w_k[-2], w_k[-1]], [0, y(w_k[-1])], color="black", linewidth=0.5, linestyle="--")
plt.scatter(w_k, np.zeros_like(w_k), marker="o", c='none', edgecolors='black', s=10)
plt.scatter(w_k, y(w_k), marker="o", c='none', edgecolors='black', s=10)
plt.plot(x, y(x), color="black", linewidth=0.5)
plt.plot(x, np.zeros_like(x), color="black", linewidth=0.5)
plt.show()

print("result: ", w_k[-1])