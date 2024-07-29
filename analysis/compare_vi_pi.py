import os
from plot import PlotStateValue


if __name__ == "__main__":
    root = os.getcwd() + "/result/"
    files = [root + "value_iteration/state_value_history.txt",
             root + "policy_iteration/state_value_history(truncate=1).txt",
             root + "policy_iteration/state_value_history(truncate=2).txt",
             root + "policy_iteration/state_value_history(truncate=5).txt",
             root + "policy_iteration/state_value_history(truncate=100).txt",
             root + "policy_iteration/state_value_history(truncate=1000).txt"]
    labels = ["vi", "pi(truncate=1)", "pi(truncate=2)",
              "pi(truncate=5)", "pi(truncate=100)", "pi(truncate=1000)"]

    plot = PlotStateValue(state_value_files=files, labels=labels)
    plot.show()
