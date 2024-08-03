import matplotlib.pyplot as plt
from typing import Optional


class PlotStateValue(object):
    def __init__(self, k_range=50, state_value_files: Optional[list]=None, labels: Optional[list]=None):
        self.k = range(k_range)
        self.labels = labels
        self.state_value_list = []
        for file_name in state_value_files:
            with open(file_name, "r") as f:
                self.state_value_list.append(list(map(lambda line: float(line.strip()), f.readlines()[: k_range])))
    
    def show(self):
        for state_value_history, label in zip(self.state_value_list, self.labels):
            plt.plot(self.k, state_value_history, label=label)
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("state value")
        plt.show()


class RecordStateValue(object):
    def __init__(self, file_name):
        self.file = open(file_name, "w")
        self.state_value_history = ["0\n"]
    
    def add(self, state_value):
        self.state_value_history.append(f"{state_value}\n")

    def __del__(self):
        self.file.writelines(self.state_value_history)
        self.file.close()


class Recorder(object):
    def __init__(self, file_name):
        self.file = open(file_name, "w")
        self.record = []
    
    def add(self, data):
        self.record.append(f"{data}\n")

    def __del__(self):
        self.file.writelines(self.record)
        self.file.close()

