import json as js
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import math as mt



#функция для визуализации графкика распределения роста и веса
def plots():
        with open("data_set.json", "r", encoding=("utf-8")) as file:
            data = js.load(file)
            height_data = []
            weight_data = []
            names = []
            for index in range(len(data)):
                height_data.append(data[str(index)]["Height"])
                weight_data.append(data[str(index)]["Weight"])
                names.append(data[str(index)]["Name"])
            height_data = np.array(height_data)
            weight_data = np.array(weight_data)
            
            
        figure, axes = plt.subplots()
        
        axes.plot(range(0, len(height_data)), height_data, label="рост", color="blue")
        axes.plot(range(0, len(weight_data)), weight_data, label="вес", color="red")
        plt.legend(loc="upper left")
        axes.set_title("Рост и вес пользователей")
        axes.set_xticks(range(0, len(names)), names, rotation=45)

        """axes[1].hist(weight_data, len(weight_data), color="red", label="гистограмма по весу")
        axes[1].hist(height_data, len(height_data), color="green", label="гистрограмма по росту")
        plt.legend(loc="upper left")
        axes[1].set_title("данные о росте и весе в виде гистрограммы")
        axes[1].set_xticks(range(0, len(names)), names, rotation=45)

        axes[2].plot(range(0, len(height_data)), np.log(height_data), color="red", label="log[рост]")
        axes[2].plot(range(0, len(weight_data)), np.log(weight_data), color="blue", label="log[вес]")
        axes[2].legend(loc="upper left")
        axes[2].set_title("данные о росте и весе в виде логорифмической функции")
        axes[2].set_xticks(range(0, len(names)), names, rotation=45)"""



#функция визуализации данных в виде гистограммы     
def histo():
     with open("data_set.json", "r", encoding=("utf-8")) as file:
            data = js.load(file)
            height_data = []
            weight_data = []
            names = []
            for index in range(len(data)):
                height_data.append(data[str(index)]["Height"])
                weight_data.append(data[str(index)]["Weight"])
                names.append(data[str(index)]["Name"])
            height_data = np.array(height_data)
            weight_data = np.array(weight_data)
            figure, axes = plt.subplots()
            axes.hist(weight_data, len(weight_data), color="red", label="гистограмма по весу")
            axes.hist(height_data, len(height_data), color="green", label="гистрограмма по росту")
            plt.legend(loc="upper left")
            axes.set_title("данные о росте и весе в виде гистрограммы")
            axes.set_xticks(range(0, len(names)), names, rotation=45)

#фукнция для визуализации данных распределения роста и веса в виде логорифмической функции
def log():
     with open("data_set.json", "r", encoding=("utf-8")) as file:
            data = js.load(file)
            height_data = []
            weight_data = []
            names = []
            for index in range(len(data)):
                height_data.append(data[str(index)]["Height"])
                weight_data.append(data[str(index)]["Weight"])
                names.append(data[str(index)]["Name"])
            height_data = np.array(height_data)
            weight_data = np.array(weight_data)
            figure, axes = plt.subplots()
            axes.plot(range(0, len(height_data)), np.log(height_data), color="red", label="log[рост]")
            axes.plot(range(0, len(weight_data)), np.log(weight_data), color="blue", label="log[вес]")
            axes.legend(loc="upper left")
            axes.set_title("данные о росте и весе в виде логорифмической функции")
            axes.set_xticks(range(0, len(names)), names, rotation=45)
     

#функция для генерации данных в формате json
def generate_data(units: int, cmap: str):

    data_set = {}

    for index in range(0, units):
        height_data = rd.gauss(180, sigma=17.70)
        weight_data = rd.gauss(79.0, sigma=17.70)
        data_set[str(index)] = {
            "Name": f"None{index}",
            "Height": height_data,
            "Weight": weight_data
        }
    
    with open("data_set.json", "w", encoding=("utf-8")) as file:
        js.dump(data_set, file)



generate_data(units=1000, cmap="winter")
plots()
plt.show()

