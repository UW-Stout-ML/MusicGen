import matplotlib.pyplot as plt
import json
import numpy as np

with open('counts.json', 'r') as file:
    data = json.load(file)
    data = np.array(data)

    plt.plot(data[1:])
    plt.show()