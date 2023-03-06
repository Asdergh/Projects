from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animo
import random as rd
import json as js

# класс StandardScaler позволяет пользоваться методами предобработки данных для оптимизации нейронной сети
# но в данном проекте использоваться будет отлько метод стадартизации данных: [std[] = (elem from undo_array) - mean(undo_array) / std(undo_array)]
# тут mean - среднее между элементами вектора undo_array
# std - стандартное отклонение между элементами undo_array

class Perceptron():
    def __init__(self, eta=0.01, epochs=40) -> None:
        self.eta = eta
        self.epochs = epochs
    
    def fit(self, X, y):
        self.W_ = np.zeros(1 + X.shape[1])
        self.error_ = []
        for _ in range(self.epochs):
            error = 0
            for (xi, true_label) in zip(X, y):
                update = true_label - self.prediction(xi)
                self.W_[1:] += self.eta * update * xi
                self.W_[0] += self.eta * update
                error += int(update != 0.0)
            self.error_.append(error)
        return self

    def pure_input(self, X):
        return np.dot(X, self.W_[1:]) + self.W_[0]
    
    def prediction(self, X):
        return np.where(self.pure_input(X) >= 0.0, 1, -1)

#реализация модели машинного обучения на основе регресивного спуска
#данная модель основанная на минимизации функции стоимости: J(w) = 1/2 * eta * (true_label - pure_input).sum() * xi
#обновление весов проиходит на основе данной функции: delta(w) = 1/2 * eta * (true_label) * X

class Regresion():
    def __init__(self, eta=0.01, epochs=40, shuffle=True) -> None:
        self.eta = eta
        self.epochs = epochs
        self.shuffle = shuffle
    
    def fit(self, X, y):
        self.W_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for _ in range(self.epochs):
            if self.shuffle == True:
                self.shuffle_(X, y)
            output = self.pure_input(X)
            error = (y - output)
            self.W_[1:] += self.eta * X.T.dot(error)
            self.W_[0] += self.eta * error.sum()
            cost = (error ** 2).sum() / 2
            self.cost_.append(cost)
        self.cost_ = np.array(self.cost_)
        return self
    
    def shuffle_(self, X, y):
        rand = np.random.permutation(len(y))
        return X[rand], y[rand]
    
    def pure_input(self, X):
        return np.dot(X, self.W_[1:]) + self.W_[0]
    
    def prediction(self, X):
        return np.where(self.pure_input(X) >= 0.0, 1, -1)

#данная функция генерирует данные и формирует из них json файл
#данные распределяются на основе гауссовского шума
def generate_json_base(units: int) -> None:
    data = {}
    for elem in range(units):
        mean_height_women = rd.gauss(163.3, sigma=7.79)
        mean_weight_women = rd.gauss(72.7, sigma=7.79)
        mean_height_men = rd.gauss(175.4, sigma=7.79)
        mean_weight_men = rd.gauss(76.7, sigma=7.79)
        random_gender = rd.choice(["Female", "Male"])
        if random_gender == "Female":
            data[str(elem)] = {

                    "Name_1": f"None_1{elem}",
                    "Gender": random_gender,
                    "Height": mean_height_women,
                    "Weight": mean_weight_women,
                }
        else:
            data[str(elem)] = {

                    "Name_1": f"None_1{elem}",
                    "Gender": random_gender,
                    "Height": mean_height_men,
                    "Weight": mean_weight_men,
                }

        
    
    with open("female_or_male.json", "w") as file:
        js.dump(data, file)

 #данная функция подгружает данные с нашего json data сета
def load_dataset():
    with open("female_or_male.json", "r") as file:
        data = js.load(file)
        X_1, X_2 = [], []
        y = []
        for i in range(len(data)):
            X_1.append(data[str(i)]["Height"])
            X_2.append(data[str(i)]["Weight"])
            y.append(data[str(i)]["Gender"])
        X_1, X_2 = np.array(X_1), np.array(X_2)
        X = np.stack((X_1, X_2))
        X = X.T
        y = np.array(y)
        y = np.where(y=="Female", 1, -1)
    return X, y



generate_json_base(units=100)
X, y = load_dataset()
#делим наш датасет при помощи метода train_test_split 
#тут test_size - есть соотношение в котором мы хотим поделить наш датасет (в данном случае 30% на X_train и 70% на X_test)
#a random_staet - это диапазон значений для перемещивания датасета
X_train, X_test, y_trian, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# дальше после того как мы разделили наш датасет применяем к нему метод стандартизации описанный ранне при помощи метода transfrom класса StandardScaler
# метод fit вычисляет нужный для стандартизации параметры такие как среднее из исходного вектора и стандартное отклонение из исходного вектора
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
ppn = Perceptron(eta=0.01, epochs=50).fit(X_train, y_test)
ada = Regresion(eta=0.01, epochs=50).fit(X_train, y_trian)
figure, axes = plt.subplots(ncols=2, figsize=(9, 9))
axes[0].scatter(x=X[:, 0], y=X[:, 1], c=y, cmap="coolwarm", s=0.98)
axes[1].plot(range(0, len(ppn.error_)), np.log(ppn.error_), color="red", label="errors in ppn")
axes[1].plot(range(0, len(ada.cost_)), np.log(ada.cost_), color="blue", label="errors in ada")
output_prediction_ppn = ppn.prediction(X_test_std)
output_prediction_ada = ada.prediction(X_test_std)
print(f"число правильно отгадоных людей от ada: {(y_test == output_prediction_ada).sum()}")
print(f"число правильно отгадоных людей от ppn: {(y_test == output_prediction_ppn).sum()}")
plt.title("log[error]")

plt.show()






    