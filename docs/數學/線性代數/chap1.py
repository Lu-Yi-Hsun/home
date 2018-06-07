import numpy as np  
import matplotlib.pyplot as plt  
def graph(formula, x_range):  
    x = np.array(x_range)  
     
    for f in formula:
        y = eval(f)
        plt.plot(x, y,2)
    plt.axhline(y=0, c='black')
    plt.axvline(x=0, c='black')
    plt.show()

line=["2*x","x/2+3/2"]
graph(line, range(-100, 100))