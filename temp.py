import matplotlib.pyplot as plt
import pandas as pd

def plot_memory(file):
    data = pd.read_csv(file, sep='\t')
    print(data.iloc[:, 1]*10**4)
    plt.plot(range(len(data.iloc[:, 0])), data.iloc[:, 0], label='mem')
    plt.plot(range(len(data.iloc[:, 1])), data.iloc[:, 1]*10**7.5, label='cells')
    plt.legend()
    plt.show()

plot_memory(r'C:\Users\php23rjb\Downloads\temp\memory.txt')