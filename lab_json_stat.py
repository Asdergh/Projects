import numpy as np
import matplotlib.pyplot as plt
import random as rd

class RandomSEt:
    def __init__(self, count_of_points, max_distance) -> None:
        self.points = count_of_points
        self.max_walk = max_distance
        self.data = []
    
    def show(self):
        data = self.data
        X, Y, Z = [], [], []

        for (x, y, z) in data:
            X.append(x)
            Y.append(y)
            Z.append(z)
        X, Y, Z = np.array(X), np.array(Y), np.array(Z)
        axes_3d = plt.figure().add_subplot(projection="3d")
        axes_3d.set_xlabel("X")
        axes_3d.set_ylabel("Y")
        axes_3d.set_zlabel("z")
        axes_3d.scatter(X, Y, Z, c=Z, cmap="magma", s=0.98)
        axes_3d.plot(X, Y, Z, color="black", linewidth=0.23)
        plt.show()
        

    def random_set(self):
        data = []
        x_grid = [0]
        y_grid = [0]
        z_grid = [0]
        for elem in range(self.points):

            x_step = rd.randint(0, 100)
            x_direction = rd.choice([-1, 1])
            x = x_grid[-1] + x_step * x_direction

            y_step = rd.randint(0, 100)
            y_direction = rd.choice([-1, 1])
            y = y_grid[-1] + y_step * y_direction

            z_step = rd.randint(0, 100)
            z_direction = rd.choice([-1, 1])
            z = z_grid[-1] + z_step * z_direction

            x_grid.append(x)
            y_grid.append(y)
            z_grid.append(z)
        
        for core in zip(x_grid, y_grid, z_grid):
            data.append(core)
        
        data = np.array(data)
        self.data = data
        return data
    
    """def generate_random_walk(self):
        x_grid = [rd.randint(0, 123)]
        y_grid = [rd.randint(0, 123)]
        z_grid = [rd.randint(0, 123)]

        for elem in range(self.points):
            tmp_list = []
            for core in range(self.max_walk):"""


    
if __name__ == "__main__":
    new_object = RandomSEt(1000, 5)
    new_object.random_set()
    new_object.show()




