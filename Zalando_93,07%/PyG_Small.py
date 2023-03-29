import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import matplotlib.pyplot as plt
from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras import Model
from keras.optimizers import RMSprop
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import generacionPoblacion as gP
import funcionesGenetico as fG
import os

# Genetico estacionario
def geneticoTFG(train_dir,validation_dir,test_dir):
    nPoblacion = 10
    # Tamaño del torneo 30% poblacion
    k = (nPoblacion * 0.3).__int__()

    [poblacion, puntuacion] = gP.generaPoblacionPyG(nPoblacion)

    # Comienza el genetico
    i = 0
    iteracionesMax = 20

    noMejoras = 0
    noMejorasMax = 5

    evaluaciones = nPoblacion  # Evaluaciones es igual al numero de poblacion que haya
    evaluacionesMaximas = 400
    mediaPuntAnt = 0

    while evaluaciones < evaluacionesMaximas:  # Para si se lleva mas de 5 iteraciones sin mejorar la media
        print(f"\n-------------------------\n-----Iteración {i} :-----\n-------------------------")
        print(f"\n-------------------------\n-----Evaluaciones {evaluaciones} :-----\n-------------------------")
        # Seleccion por torneo
        padre1 = fG.torneo(puntuacion, k)
        padre2 = fG.torneo(puntuacion, k)
        while padre2 == padre1:
            padre2 = fG.torneo(puntuacion, k)
        print(f"\t--> Padres obtenidos del torneo")
        print(f"\t\t Padre 1: {poblacion[padre1]}")
        print(f"\t\t Padre 2: {poblacion[padre2]}")
        # Cruzo a los padres
        hijo1, hijo2 = fG.cruce2(poblacion[padre1], poblacion[padre2])
        print(f"\t--> Tras cruzar a los padres obtengo")
        print(f"\t\t Hijo 1: {hijo1}")
        print(f"\t\t Hijo 2: {hijo2}")
        # Valido los hijos a continuacion
        hijo1 = gP.traduccionValida(hijo1)
        hijo2 = gP.traduccionValida(hijo2)
        # Muto a los hijos
        print(f"\t--> Muto a los hijos")
        mutado1 = fG.mutacion(hijo1)
        mutado2 = fG.mutacion(hijo2)
        # Ya se validan los hijos dentro de la mutacion, luego no tengo que volver a hacerlo
        print(f"\t\t Mutado 1: {mutado1}")
        print(f"\t\t Mutado 2: {mutado2}")
        '''mutado1 = gP.traduccionValida(mutado1)
        mutado2 = gP.traduccionValida(mutado2)'''

        # En caso de configuraciones fallidas o imposibles, las castiga mucho
        try:
            puntHijo1 = fG.puntuacionCromosoma_PyG(mutado1,train_dir,validation_dir,test_dir)
        except:
            puntHijo1 = 0
        try:
            puntHijo2 = fG.puntuacionCromosoma_PyG(mutado2,train_dir,validation_dir,test_dir)
        except:
            puntHijo2 = 0

        evaluaciones += 2

        print(f"Puntuacion 1: {puntHijo1}")
        print(f"Puntuacion 2: {puntHijo2}")
        # Reemplazo
        poblacion, puntuacion = fG.reemplazo(poblacion, puntuacion, [mutado1, mutado2], [puntHijo1, puntHijo2])
        print("\n\t\t--------Poblacion: ---------")
        for k in range(len(poblacion)):
            print(f"Individuo {k}: {puntuacion[k]} || {poblacion[k]} ")
        mediaPunt = sum(puntuacion) / len(puntuacion)

        if mediaPunt <= mediaPuntAnt:  # Si no mejora
            noMejoras += 1
        else:
            noMejoras = 0
        i += 1

# Genetico generacional 1 elite
def geneticoTFG_Gen(train_dir,validation_dir,test_dir):
    nPoblacion = 10
    # Elijo 3 para el torneo
    k = 3

    poblacionInicial = 10
    [poblacion, puntuacion] = gP.generaPoblacionPyG(poblacionInicial,train_dir,validation_dir,test_dir)

    tamElite = 2

    i = 0

    tamActual = 0
    poblacionNueva = []
    puntuacionNueva = []

    evaluaciones = poblacionInicial
    evaluacionesMaximas = 500

    listaEvaluaciones = []
    listaPuntuaciones = []
    listaEvaluaciones.append(0)

    indiceMejorPuntuacion = np.argmax(puntuacion)

    listaPuntuaciones.append(puntuacion[indiceMejorPuntuacion])

    print("\n\t\t--------Poblacion: ---------")
    for k in range(len(poblacion)):
        print(f"Individuo {k}: {puntuacion[k]} || {poblacion[k]} ")
    print(f"Media puntos: {sum(puntuacion) / len(puntuacion)}")

    while evaluaciones < evaluacionesMaximas:
        print(f"\n-------------------------\n-----Iteración {i} :-----\n-------------------------")
        print(f"\n-------------------------\n-----Evaluaciones {evaluaciones} :-----\n-------------------------")

        # Seleccion por torneo
        padre1 = fG.torneo(puntuacion, k)
        padre2 = fG.torneo(puntuacion, k)
        while padre2 == padre1:
            padre2 = fG.torneo(puntuacion, k)
        # Cruzo a los padres
        hijo1, hijo2 = fG.cruce2(poblacion[padre1], poblacion[padre2])
        # Valido los hijos a continuacion
        hijo1 = gP.traduccionValida(hijo1)
        hijo2 = gP.traduccionValida(hijo2)
        # Muto a los hijos
        mutado1 = fG.mutacion(hijo1)
        mutado2 = fG.mutacion(hijo2)
        #Evaluo a los hijos
        try:
            puntHijo1 = fG.puntuacionCromosoma_PyG(mutado1, train_dir, validation_dir, test_dir)
        except:
            puntHijo1 = 0
        try:
            puntHijo2 = fG.puntuacionCromosoma_PyG(mutado2, train_dir, validation_dir, test_dir)
        except:
            puntHijo2 = 0
        #Añado +2 a las evaluaciones
        evaluaciones += 2
        hijos = [mutado1, mutado2]
        puntHijos = [puntHijo1, puntHijo2]
        # Añado individuos a la población
        if (tamActual + 2) <= nPoblacion:
            poblacionNueva.append(mutado1)
            poblacionNueva.append(mutado2)

            puntuacionNueva.append(puntHijo1)
            puntuacionNueva.append(puntHijo2)
        # Muestra a la poblacion
        print("\n\t\t--------Poblacion: ---------")
        for k in range(len(poblacion)):
            print(f"Individuo {k}: {puntuacion[k]} || {poblacion[k]} ")
        print(f"Media puntos: {sum(puntuacion) / len(puntuacion)}")

        # Me quedo con la nueva poblacion
        if len(poblacionNueva) == nPoblacion:
            # Guardo al mejor de la poblacion y lo sustituyo por el peor
            # Obtengo el mejor de la actual
            mejorIndice = np.argmax(puntuacion)

            mejorCromosoma = poblacion[mejorIndice]
            mejorPuntuacion = puntuacion[mejorIndice]
            # Lo sustituyo por el peor de la nueva
            peorIndice = np.argmin(puntuacionNueva)

            poblacionNueva[peorIndice] = mejorCromosoma
            puntuacionNueva[peorIndice] = mejorPuntuacion

            poblacion = poblacionNueva
            puntuacion = puntuacionNueva

            poblacionNueva = []
            puntuacionNueva = []

            indiceMejorPuntuacion = np.argmax(puntuacion)
            print(f"Añado: {puntuacion[indiceMejorPuntuacion]}")
            listaPuntuaciones.append(puntuacion[indiceMejorPuntuacion])
            listaEvaluaciones.append(evaluaciones)
            tamActual = 0


        i += 1
    plt.plot(listaEvaluaciones, listaPuntuaciones)
    plt.show()

# Genetico generacional 2 elite
def geneticoTFG_Gen2(train_dir,validation_dir,test_dir):
    nPoblacion = 10
    # Elijo 3 para el torneo
    k = 3

    poblacionInicial = 10
    [poblacion, puntuacion] = gP.generaPoblacionPyG(poblacionInicial,train_dir,validation_dir,test_dir)

    tamElite = 2

    i = 0

    tamActual = 0
    poblacionNueva = []
    puntuacionNueva = []

    evaluaciones = poblacionInicial
    evaluacionesMaximas = 500

    listaEvaluaciones = []
    listaPuntuaciones = []
    listaEvaluaciones.append(0)

    indiceMejorPuntuacion = np.argmax(puntuacion)

    listaPuntuaciones.append(puntuacion[indiceMejorPuntuacion])

    print("\n\t\t--------Poblacion: ---------")
    for k in range(len(poblacion)):
        print(f"Individuo {k}: {puntuacion[k]} || {poblacion[k]} ")
    print(f"Media puntos: {sum(puntuacion) / len(puntuacion)}")

    while evaluaciones < evaluacionesMaximas:
        print(f"\n-------------------------\n-----Iteración {i} :-----\n-------------------------")
        print(f"\n-------------------------\n-----Evaluaciones {evaluaciones} :-----\n-------------------------")

        # Seleccion por torneo
        padre1 = fG.torneo(puntuacion, k)
        padre2 = fG.torneo(puntuacion, k)
        while padre2 == padre1:
            padre2 = fG.torneo(puntuacion, k)
        # Cruzo a los padres
        hijo1, hijo2 = fG.cruce2(poblacion[padre1], poblacion[padre2])
        # Valido los hijos a continuacion
        hijo1 = gP.traduccionValida(hijo1)
        hijo2 = gP.traduccionValida(hijo2)
        # Muto a los hijos
        mutado1 = fG.mutacion(hijo1)
        mutado2 = fG.mutacion(hijo2)
        #Evaluo a los hijos
        try:
            puntHijo1 = fG.puntuacionCromosoma_PyG(mutado1, train_dir, validation_dir, test_dir)
        except:
            puntHijo1 = 0
        try:
            puntHijo2 = fG.puntuacionCromosoma_PyG(mutado2, train_dir, validation_dir, test_dir)
        except:
            puntHijo2 = 0
        #Añado +2 al contador de evaluaciones
        evaluaciones += 2
        hijos = [mutado1, mutado2]
        puntHijos = [puntHijo1, puntHijo2]

        # Añado individuos a la población
        if (tamActual + 2) <= nPoblacion:
            poblacionNueva.append(mutado1)
            poblacionNueva.append(mutado2)

            puntuacionNueva.append(puntHijo1)
            puntuacionNueva.append(puntHijo2)
            tamActual += 2
        # Muestra a la poblacion
        print("\n\t\t--------Poblacion: ---------")
        for k in range(len(poblacion)):
            print(f"Individuo {k}: {puntuacion[k]} || {poblacion[k]} ")
        print(f"Media puntos: {sum(puntuacion) / len(puntuacion)}")

        # Me quedo con la nueva poblacion
        if len(poblacionNueva) == nPoblacion:
            # Guardo a los 2 mejores de la poblacion y los sustituyo por los 2 peores
            # Obtengo el mejor de la actual
            mejoresIndices = np.flip(np.argsort(puntuacion))[0:2]

            mejoresCromosomas = []
            mejoresPuntuaciones = []
            for i in range(len(mejoresIndices)):
                mejoresCromosomas.append(poblacion[mejoresIndices[i]])
                mejoresPuntuaciones.append(puntuacion[mejoresIndices[i]])
            # Lo sustituyo por los peores de la nueva
            peoresIndices = np.argsort(puntuacionNueva)[0:2]

            for j in range(len(mejoresIndices)):
                poblacionNueva[peoresIndices[j]] = mejoresCromosomas[j]
                puntuacionNueva[peoresIndices[j]] = mejoresPuntuaciones[j]

            poblacion = poblacionNueva
            puntuacion = puntuacionNueva

            poblacionNueva = []
            puntuacionNueva = []

            indiceMejorPuntuacion = np.argmax(puntuacion)
            print(f"Añado: {puntuacion[indiceMejorPuntuacion]}")
            listaPuntuaciones.append(puntuacion[indiceMejorPuntuacion])
            listaEvaluaciones.append(evaluaciones)
            tamActual = 0
        # poblacion, puntuacion = fG.reinicializo(poblacion, puntuacion, tamElite)
        i += 1
    plt.plot(listaEvaluaciones, listaPuntuaciones)
    plt.show()

if __name__ == '__main__':
    # La BD 1 es la de Zalando, mientras que la 0 es la de los números de MNIST; la 2 y 3 son las de PyG
    fG.genetico_CHC(baseDatos=2, nPoblacion=10, numeroElite=2, numeroEvaluaciones=300,
                             distanciaHinicial=5, pinta=True)
