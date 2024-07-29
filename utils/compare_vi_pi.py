import matplotlib.pyplot as plt

k = range(50)
with open("../result/value_iteration/state_value_history.txt") as f:
    vi = list(map(lambda line: float(line.strip()), f.readlines()[:len(k)]))
with open("../result/policy_iteration/state_value_history(truncate=1).txt") as f:
    pi_truncate_1 = list(map(lambda line: float(line.strip()), f.readlines()[:len(k)]))
with open("../result/policy_iteration/state_value_history(truncate=2).txt") as f:
    pi_truncate_2 = list(map(lambda line: float(line.strip()), f.readlines()[:len(k)]))
with open("../result/policy_iteration/state_value_history(truncate=5).txt") as f:
    pi_truncate_5 = list(map(lambda line: float(line.strip()), f.readlines()[:len(k)]))
with open("../result/policy_iteration/state_value_history(truncate=1000).txt") as f:
    pi_truncate_1000 = list(map(lambda line: float(line.strip()), f.readlines()[:len(k)]))


plt.plot(k, vi, '--*r', label="vi")
plt.plot(k, pi_truncate_1, '--', label="pi(truncate=1)")
plt.plot(k, pi_truncate_2, '--', label="pi(truncate=2)")
plt.plot(k, pi_truncate_5, '--', label="pi(truncate=5)")
plt.plot(k, pi_truncate_1000, '-.+g', label="pi(truncate=1000)")
plt.xlabel("k")
plt.ylabel("state value")
plt.legend()
plt.show()