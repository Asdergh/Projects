import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import json as js


with open("data.json", "r", encoding="utf-8") as file:


    data = js.load(file)

"""def array_sorter(array_to_sort):
    for i in range(0, len(array_to_sort)):
        for j in range(i, 0, -1):
            if array_to_sort[j] > array_to_sort[i]:
                array_to_sort[j], array_to_sort[i] = array_to_sort[i], array_to_sort[j]
    
    return array_to_sort"""

height_data = []
weight_data = []
names = []

for iter in range(1, len(data)):
    
    height_data.append(int(data[str(iter)]["Heigth"]))
    weight_data.append(int(data[str(iter)]["Weight"]))
    names.append(data[str(iter)]["Name"])

sorted_height = sorted(height_data)
sorted_weight = sorted(weight_data)

mean_value_height = stat.mean(height_data)
mean_value_weight = stat.mean(weight_data)

spread_of_height = stat.variance(height_data) 
spread_of_weight = stat.variance(weight_data)

spread_h = stat.variance(sorted_height)
spread_w = stat.variance(sorted_weight)

figure, axes_2d = plt.subplots()

axes_2d.scatter(range(0, len(height_data)), height_data, marker="s", color="green")
axes_2d.plot(range(0, len(height_data)), height_data, color="green", label=f"данные о росте: me: {mean_value_height}, sp: {spread_of_height}")

axes_2d.scatter(range(0, len(weight_data)), weight_data, marker="^", color="red")
axes_2d.plot(range(0, len(weight_data)), weight_data, color="red", label=f"данные о весе: me{mean_value_weight}, sp{spread_of_weight}")

axes_2d.scatter(range(0, len(sorted_weight)), sorted_weight, marker="o", color="blue")
axes_2d.plot(range(0, len(sorted_weight)), sorted_weight, label=f"{spread_w}", color="blue")

axes_2d.scatter(range(0, len(sorted_height)), sorted_height, marker="v", color="y")
axes_2d.plot(range(0, len(sorted_height)), sorted_height, label=f"{spread_h}", color="y")
axes_2d.legend(loc="upper left")

plt.xticks(range(0, len(names)), names, rotation=45)
plt.show()

