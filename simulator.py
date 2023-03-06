import matplotlib.pyplot as plt
import matplotlib.animation as animo
import pandas as pd
import numpy as np
import random as rd


figure = plt.figure()
axes_3d = figure.add_subplot(projection="3d")
print(plt.style.available)
plt.style.use(["dark_background"])
#создадим функцию обработки данных и форматирования их под формат нужный для создания графика


def generate_data(units: int) -> None:

    with open("data_file.txt", "w") as file:
        X = np.linspace(-np.pi, np.pi, units)
        Y = np.linspace(-np.pi, np.pi, units)
        Z = np.linspace(-np.pi, np.pi, units)
        for index in range(len(X)):
            file.write(f"{X[index]}\t{Y[index]}\t{Z[index]}\n")

time_inter = np.linspace(0, 20, 100)
def anim(i):
        u = np.linspace(-np.pi, np.pi, 100)
        v = np.linspace(-np.pi, np.pi, 100)
        u, v = np.meshgrid(u, v)
        
        Z = np.sin(u + time_inter[i]) / np.cos(v + time_inter[i])
    
        axes_3d.clear()
        axes_3d.plot_surface(u, v, Z, cmap="coolwarm", alpha=0.58)
        
        plt.title("Anim example")

def anim_2(i):
     with open("data_file.txt", "r") as file:
          
        data = file.readlines()
        core_array = []
        for core in data:
               
            point = core.split("\t")
            point_tmp = [float(x) for x in point if x.isdigit()]
            core_array.append(point_tmp)
        print(core_array)

        core_array = np.array(core_array)
        phi, theta= core_array[:, 0], core_array[:, 1]
        phi, theta = np.meshgrid(phi, theta)
        
        X_grid = np.cos(phi * time_inter[i]) * np.sin(theta * time_inter[i])
        Y_grid = np.sin(phi * time_inter[i]) * np.sin(theta * time_inter[i])
        Z_grid = np.cos(phi)

        axes_3d.plot_surface(X_grid, Y_grid, Z_grid, cmap="coolwarm", alpha=0.56)
        plt.title("Anim Surface")
        
     


generate_data(units=100)
ani = animo.FuncAnimation(figure, anim_2, interval=50)
plt.show()

