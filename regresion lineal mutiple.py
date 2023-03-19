import numpy as np
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generador(coeficientes, muestras, desvest):
    """se cuenta el número de predictores y se crea una matriz con los coeficientes
      de p filas y 1 columna para hacer la multiplicacion de matrices"""
    n_coeficientes = len(coeficientes)
    coef_matriz = np.array(coeficientes).reshape(n_coeficientes, 1)
    """X es una matriz de n muestras * n, coeficientes, y coef_matrz es
    n_coeficientes*1
    Se multiplican las matrices para obtener 
    Y dados x1 y x2 y se calcula la transpuesta para usar la regresion"""
    x = np.random.random_sample((muestras, n_coeficientes))*100
    epsilon = np.random.randn(muestras)*desvest
    y = np.matmul(x, coef_matriz).transpose()+epsilon
    return x, y

coeficientes_reales = [10, 5]
muestras = 200
desvest = 100
#Llamamos a la funcion para generar matriz
X, Y = generador(coeficientes_reales, muestras, desvest)

#Creamos el modelo
modelo = linear_model.LinearRegression()

#Entrenamos el modelo
modelo.fit(X, Y.transpose())
print("Coeficientes: ", modelo.coef_[0])

#Predecimos
y_pred = modelo.predict(X)
print("Error cuadrático medio: ", mean_squared_error(Y.transpose(), y_pred))
print("R2:", r2_score(Y.transpose(), y_pred))

fig, [plot1, plot2] = plt.subplots(1,2)
fig.set_figwidth(15)

plot1.scatter(X[:,0], Y)
plot1.set_title("Y en función de X1")
plot1.set_xlabel('X1')
plot1.set_ylabel('Y')


plot2.scatter(X[:,1], Y)
plot2.set_title("Y en función de X2")
plot2.set_xlabel('X2')
plot2.set_ylabel('Y')

plot3=plt.figure(figsize = (15, 10)).gca(projection = '3d')
X1, X2 = np.meshgrid(range(100), range(100))
z_modelo = modelo.coef_[0][0]*X1+modelo.coef_[0][1]*X2
z_real = coeficientes_reales[0]*X1+coeficientes_reales[1]*X2
plot3.plot_surface(X1,X2, z_modelo, alpha = 0.3, color = 'yellow')
plot3.plot_surface(X1, X2, z_real, alpha = 0.3, color = 'red')
plot3.scatter(X[:,0], X[:,1], Y)
plot3.set_title("Regresión lineal con dos variables")
plot3.set_xlabel('Eje X1')
plot3.set_ylabel('Eje X2')
plot3.set_zlabel('Eje Y')
plot3.view_init(10, )
plt.show()









