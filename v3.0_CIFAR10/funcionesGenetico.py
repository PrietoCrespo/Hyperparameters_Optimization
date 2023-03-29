import random
import variables as v
import tensorflow as tf
'''from tensorflow.keras.optimizers import RMSprop
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras'''
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import numpy as np
import generacionPoblacion
import generacionPoblacion as gP

#TODO en el relleno con padre, el tamaño del hijo no tiene sentido, demasiado chico.

inicioConv = 3
inicioPooling = inicioConv + 4 * v.getNumCapasConvReal()[-1]
inicioDensas = inicioPooling + v.getNumCapasPooling()[-1]


def distanciaHamming(cromosoma1, cromosoma2):
    cromosoma1 = list(cromosoma1.copy())
    cromosoma2 = list(cromosoma2.copy())

    cromosoma1 = arrayLimpio(cromosoma1)
    cromosoma2 = arrayLimpio(cromosoma2)

    if len(cromosoma1) > len(cromosoma2):
        for i in range(len(cromosoma1) - len(cromosoma2)):
            cromosoma2.append(0)
    elif len(cromosoma2) > len(cromosoma1):
        for i in range(len(cromosoma2) - len(cromosoma1)):
            cromosoma1.append(0)
    hamming_distance = (hamming(cromosoma1, cromosoma2) * len(cromosoma1)).__int__()

    return hamming_distance


def rellenoConPadre(hijo, padre):
    numCapasConvReal = v.getNumCapasConvReal()
    numCapasConv = v.getNumCapasConv()
    numCapasPooling = v.getNumCapasPooling()
    numCapasDensas = v.getNumCapasDensas()
    numCapasDensasReal = v.getNumCapasDensasReal()
    tamFiltros = v.getTamFiltrosReal()
    numFiltros = v.getNumFiltrosReal()
    numFiltrosDensas = v.getNumFiltrosReal()
    stride = v.getStride()  # Pasos en que se mueve la ventana de los filtros
    padding = v.getPaddingReal()
    Factivacion = v.getFactivacionReal()
    optimizadores = v.getOptimizadoresReal()
    dropout = v.getDropoutReal()
    batchNorm = v.getBatchNormReal()

    hijoCompleto = []

    print(f"Relleno al hijo: {hijo} \ncon {padre}")
    capasConv = numCapasConvReal[hijo[0] - 1]
    capasConvNoReal = hijo[0]
    capasPooling = hijo[1]
    capasDensas = numCapasDensasReal[hijo[2] - 1]
    capasDensasNoReal = hijo[2]

    #ValoresHijo
    inicioPoolingH = (inicioConv) + (capasConv * 4)
    inicioDensasH = inicioPoolingH + capasPooling
    #Añado las capas
    hijoCompleto.append(capasConvNoReal)
    hijoCompleto.append(capasPooling)
    hijoCompleto.append(capasDensasNoReal)
    #Vector conv
    arrayConv = hijo[inicioConv:(inicioConv) + (capasConv * 4)]
    #Primero añado la capa conv del hijo
    for l in arrayConv:
        hijoCompleto.append(l)
    #Añado los valores restantes del padre de conv
    valoresRestantesPadre = padre[inicioConv + (capasConv * 4):inicioPooling]
    for l in valoresRestantesPadre:
        hijoCompleto.append(l)
    #Vector pooling
    arrayPool = hijo[inicioPoolingH:inicioPoolingH + capasPooling ]
    #Añado los valores pooling del hijo
    for l in arrayPool:
        hijoCompleto.append(l)
    #Añado los valores restantes del padre de pooling
    valoresRestantesPadre = padre[inicioPooling + capasPooling:inicioDensas]
    for l in valoresRestantesPadre:
        hijoCompleto.append(l)
    #Vector densas
    arrayDensas = hijo[inicioDensasH:inicioDensasH + (capasDensas * 4)]
    #Añado los valores de capa densa del hijo
    for l in arrayDensas:
        hijoCompleto.append(l)
    #Añado los valores restantes del padre de densas
    valoresRestantesPadre = padre[inicioDensas + capasDensas*4:-3]
    for l in valoresRestantesPadre:
        hijoCompleto.append(l)
    # Stride, opt y fAct
    stride = hijo[-3]
    opt = hijo[-2]
    fAct = hijo[-1]

    hijoCompleto.append(stride)
    hijoCompleto.append(opt)
    hijoCompleto.append(fAct)

    return hijoCompleto

def arrayLimpio(cromosoma):
    numCapasConvReal = v.getNumCapasConvReal()
    numCapasConv = v.getNumCapasConv()
    numCapasPooling = v.getNumCapasPooling()
    numCapasDensas = v.getNumCapasDensas()
    numCapasDensasReal = v.getNumCapasDensasReal()
    tamFiltros = v.getTamFiltrosReal()
    numFiltros = v.getNumFiltrosReal()
    numFiltrosDensas = v.getNumFiltrosReal()
    stride = v.getStride()  # Pasos en que se mueve la ventana de los filtros
    padding = v.getPaddingReal()
    Factivacion = v.getFactivacionReal()
    optimizadores = v.getOptimizadoresReal()
    dropout = v.getDropoutReal()
    batchNorm = v.getBatchNormReal()

    cromosomaDevuelto = []
    capasConv = cromosoma[0]
    capasConvReales = numCapasConvReal[capasConv - 1]
    capasPool = cromosoma[1]
    capasDensas = cromosoma[2]
    capasDensasReales = numCapasDensasReal[capasDensas - 1]
    #Añado las capas
    cromosomaDevuelto.append(capasConv)
    cromosomaDevuelto.append(capasPool)
    cromosomaDevuelto.append(capasDensas)
    #Vector conv
    arrayConv = cromosoma[inicioConv:(inicioConv) + (capasConvReales * 4)]
    for l in arrayConv:
        cromosomaDevuelto.append(l)
    #Vector pooling
    arrayPool = cromosoma[inicioPooling:inicioPooling + capasPool]
    for l in arrayPool:
        cromosomaDevuelto.append(l)
    #Vector densas
    arrayDensas = cromosoma[inicioDensas:inicioDensas + capasDensasReales * 4]
    for l in arrayDensas:
        cromosomaDevuelto.append(l)
    #Stride, opt y fAct
    stride = cromosoma[-3]
    opt = cromosoma[-2]
    fAct = cromosoma[-1]

    cromosomaDevuelto.append(stride)
    cromosomaDevuelto.append(opt)
    cromosomaDevuelto.append(fAct)

    return cromosomaDevuelto

# Geneticos
def genetico_CHC(nPoblacion, numeroElite, distanciaHinicial, prueba):
    imagenes = preprocesadoImagenes()
    if prueba:
        imagenes = generaTabla() #Las imagenes la pongo como tabla
    #Genero poblacion
    poblacionInicial = nPoblacion
    [poblacion, puntuacion, evaluaciones] = gP.generaPoblacion(poblacionInicial, imagenes=imagenes)
    #Para el generacional
    tamElite = numeroElite

    # Para mostrar la grafica
    listaCromosomas = []
    listaEvaluaciones = []
    listaPuntuaciones = []
    listaEvaluaciones.append(evaluaciones)
    indiceMejorPuntuacion = np.argmax(puntuacion)
    listaPuntuaciones.append(puntuacion[indiceMejorPuntuacion])
    listaCromosomas.append(poblacion[indiceMejorPuntuacion])
    mejorAnt = puntuacion[indiceMejorPuntuacion]
    #Muestro a la poblacion
    print("\n\t\t--------Poblacion tras generar individuos: ---------")
    for k in range(len(poblacion)):
        print(f"Individuo {k}: {puntuacion[k]} || {poblacion[k]} ")

    # Inicializo variables genetico
    generacion = 1 #Contando la inicial
    generacionesMaximas = 35
    reinicios = 0
    poblacionNueva = []
    puntuacionNueva = []
    distHinicial = distanciaHinicial
    distHmax = distHinicial #Inicializo la distancia hamming
    #Bucle genetico
    while generacion < generacionesMaximas:
        print(f"\n-------------------------\n-----Generacion {generacion} :-----\n-------------------------")
        print("Cruzo a todos los padres: ")
        tamActual = 0
        indices = list(range(0, len(poblacion)))
        np.random.shuffle(indices)
        print(indices)
        #Cruzo a todos los padres
        for i in range(0,len(poblacion)-1,2):
            print(f"Indice: {i}")
            indPadre1 = indices[i]
            indPadre2 = indices[i + 1]

            distanciaPadres = distanciaHamming(poblacion[indPadre1], poblacion[indPadre2])
            print(f"Padres: \n\t{poblacion[indPadre1]}\n\t{poblacion[indPadre2]}")
            print(distanciaPadres)
            #Cruzo
            if distanciaPadres > distHmax:
                padre1 = poblacion[indPadre1]
                padre2 = poblacion[indPadre2]

                if len(padre1) != len(padre2):
                    raise Exception("padres con tamaño distinto")
                hijo1, hijo2 = cruce_CHC_Alternativo(padre1, padre2)
                if len(hijo1) != len(hijo2):
                    raise Exception("Hijos con tamaño distinto")
                print(f"Tam hijo1: {len(hijo1)}")
                print(f"Tam hijo2: {len(hijo2)}")
                puntHijo1 = puntuacionCromosoma(hijo1, imagenesPasadas=imagenes)
                if puntHijo1 > 0:
                    evaluaciones += 1
                puntHijo2 = puntuacionCromosoma(hijo2, imagenesPasadas=imagenes)
                if puntHijo2 > 0:
                    evaluaciones += 1

                hijos = [hijo1, hijo2]
                puntHijos = [puntHijo1, puntHijo2]
                print(f"\n-------------------------\n-----Evaluaciones tras cruce: {evaluaciones} :-----\n-------------------------")
            else:
                print("No cruzo")
                hijos = [poblacion[indPadre1], poblacion[indPadre2]]
                puntHijos = [puntuacion[indPadre1], puntuacion[indPadre2]]

            print(f"Obtengo los hijos: \n\t{hijos[0]} | {puntHijos[0]}\n\t{hijos[1]} | {puntHijos[1]}")

            # Añado individuos a la población
            print(f"Tam actual + 2: {tamActual+2}")
            if (tamActual + 2) <= nPoblacion:
                print("Añado los individuos a la poblacion:")
                poblacionNueva.append(hijos[0])
                poblacionNueva.append(hijos[1])

                puntuacionNueva.append(puntHijos[0])
                puntuacionNueva.append(puntHijos[1])
                printPoblacion_Puntuacion(" durante el cruce", poblacionNueva, puntuacionNueva)
                tamActual += 2
        #Termino cruce de los padres, sumo 1 a la generación
        generacion += 1
        printPoblacion_Puntuacion(" tras el cruce", poblacionNueva, puntuacionNueva)
        tamCromosoma1PN = len(poblacionNueva[0])
        tamCromosoma1PP = len(poblacion[0])
        print(f"Tamaños: \n\tPoblacion nueva: {tamCromosoma1PN}\n\tPoblacion padre: {tamCromosoma1PP}")
        printPoblacion_Puntuacion(" de los padres tras cruce", poblacion, puntuacion)
        # Me quedo con los mejores entre padres e hijos
        poblacionAux = list(poblacion) + list(poblacionNueva)
        puntuacionAux = list(puntuacion) + list(puntuacionNueva)
        poblacionAnt = list(np.array(poblacion).copy()) #Me quedo con los padres
        puntuacionAnt = list(np.array(puntuacion)).copy()
        poblacion, puntuacion = selecciono_Mejores(poblacionAux,puntuacionAux, nPoblacion)

        printPoblacion_Puntuacion(" quedandome con los  mejores",poblacion,puntuacion)
        print(f"\n-------------------------\n-----Evaluaciones {evaluaciones} :-----\n-------------------------")
        # Una vez termino los cruces
        if distHmax == 0: # Reinicializo poblacion quedándome con la élite

            print("Reinicializo la poblacion, quedándome con la élite")
            poblacionReinicializada, puntuacionReinicializada = reinicializo(poblacion=poblacion,puntuacion=puntuacion,tamElite=numeroElite) #Me quedo con los k elementos mejores de la población
            reinicios += 1
            print(f"Reinicio número {reinicios}")
            printPoblacion_Puntuacion(mensaje=" al reinicializar",poblacion=poblacionReinicializada,puntuacion=puntuacionReinicializada)
            [poblacionNueva, puntuacionNueva, evaluacionesNuevas] = gP.generaPoblacion(nPoblacion - len(poblacionReinicializada), imagenes=imagenes) #Genero el resto de elementos que necesito para la poblacion total
            printPoblacion_Puntuacion(mensaje=" recien generada", poblacion=poblacionNueva, puntuacion=puntuacionNueva)
            evaluaciones += evaluacionesNuevas
            poblacion = list(poblacionReinicializada) + list(poblacionNueva)
            puntuacion = list(puntuacionReinicializada) + list(puntuacionNueva)
            printPoblacion_Puntuacion(mensaje=" + la generación nueva", poblacion=poblacion, puntuacion=puntuacion)
            distHmax = distHinicial
        else:
            #Me quedo con los mejores pero tengo que comprobar que no sea la misma poblacion que los padres, en cuyo caso tengo que disminuir la distancia max
            print("Me quedo con los mejores, pero compruebo que genero descendencia")

            mediaAnt = (sum(puntuacionAnt) / len(puntuacionAnt))
            mediaAct = sum(puntuacion) / len(puntuacion)
            if mediaAct == mediaAnt:#Es que no se ha generado descendencia
                distHmax -= 1
                print(f"Disminuyo la distancia Hamming máxima a {distHmax}")
            #En caso contrario se sigue igual


            # Muestra a la poblacion
            printPoblacion_Puntuacion("",poblacion,puntuacion)
        #Reinicio la poblacion de cruce
        poblacionNueva = []
        puntuacionNueva = []

        indiceMejorPuntuacion = np.argmax(puntuacion)

        if puntuacion[indiceMejorPuntuacion] > mejorAnt:  # Mejora
            print(f"Añado: {puntuacion[indiceMejorPuntuacion]}")
            listaPuntuaciones.append(puntuacion[indiceMejorPuntuacion])
            listaEvaluaciones.append(evaluaciones)
            listaCromosomas.append(poblacion[indiceMejorPuntuacion])
            mejorAnt = puntuacion[indiceMejorPuntuacion]
    #Añado la ultima evaluacion
    indiceMejorPuntuacion = np.argmax(puntuacion)
    listaPuntuaciones.append(puntuacion[indiceMejorPuntuacion])
    listaEvaluaciones.append(evaluaciones)
    listaCromosomas.append(poblacion[indiceMejorPuntuacion])

    print(f"Termino el CHC así con {reinicios} reinicios:")
    printPoblacion_Puntuacion(" final", poblacion, puntuacion)
    print(f"Reinicios totales: {reinicios}")
    print(f"Lista cromosomas: {listaCromosomas}")

    return listaEvaluaciones, listaPuntuaciones, reinicios,listaCromosomas



def preprocesadoImagenes():
    # Imagenes
    cifar = tf.keras.datasets.cifar10

    (train_images, train_labels), (test_images, test_labels) = cifar.load_data()

    # Hago resize de las imagenes
    train_images = train_images.reshape((50000, 32, 32, 3))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 32,32, 3))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

def redNeuronal_CNN(configuracion, imagenes):
    print(f"Red neuronal para la conf: {configuracion}")
    tamConf = len(configuracion)
    configuracion = traducirConfiguracion(configuracion).copy()
    tamConfTraducido = len(configuracion)

    numCapasConvReal = v.getNumCapasConvReal()
    numCapasConv = v.getNumCapasConv()
    numCapasPooling = v.getNumCapasPooling()
    numCapasDensas = v.getNumCapasDensas()
    numCapasDensasReal = v.getNumCapasDensasReal()
    tamFiltros = v.getTamFiltrosReal()
    numFiltros = v.getNumFiltrosReal()
    numFiltrosDensas = v.getNumFiltrosReal()
    stride = v.getStride()  # Pasos en que se mueve la ventana de los filtros
    padding = v.getPaddingReal()
    Factivacion = v.getFactivacionReal()
    optimizadores = v.getOptimizadoresReal()
    dropout = v.getDropoutReal()
    batchNorm = v.getBatchNormReal()


    # Hago resize de las imagenes
    train_images = imagenes[0]
    train_labels = imagenes[1]
    test_images = imagenes[2]
    test_labels = imagenes[3]


    capasConv = configuracion[0]
    capasPool = configuracion[1]
    capasDens = configuracion[2]
    funcion_Act = configuracion[-1]

    conf = np.array(configuracion).copy()
    #Obtengo vector posiciones pooling
    configuracionPooling = np.array(configuracion[inicioPooling:inicioPooling+capasPool]).copy()
    #Obtengo vector conv resto
    configuracionConvRest = configuracion[inicioConv + 4:(inicioConv + 4) + (capasConv * 4)]
    #Obtengo vector capas densas
    confCapasDensas = configuracion[inicioDensas:inicioDensas + (capasDens * 4)]

    print(f"Conf pooling: {configuracionPooling}")
    #Early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    # Comienzo a definir la red
    model = Sequential()
    # Valores para la primera capa
    numero_Filtros = configuracion[inicioConv]
    tamaño_Filtro = configuracion[inicioConv + 1]
    valorBatch = configuracion[inicioConv + 2]
    valorDropout = configuracion[inicioConv + 3]
    tamStrides = configuracion[-3]

    model.add(Conv2D(numero_Filtros, (tamaño_Filtro, tamaño_Filtro), padding='same', strides=tamStrides, activation=funcion_Act,
                     input_shape=(32, 32, 3)))

    if configuracionPooling.__contains__(1):
        model.add(MaxPooling2D((2, 2)))
    if valorBatch > 0:
        model.add(BatchNormalization())
    if valorDropout > 0:
        model.add(Dropout(valorDropout))


    print(f"Conf restante conv: {configuracionConvRest}")
    indPooling = 2
    # Para el resto de capas conv
    for i in range(0, len(configuracionConvRest),4):
        numero_Filtros = configuracionConvRest[i]
        tamaño_Filtro = configuracionConvRest[i + 1]
        valorBatch = configuracionConvRest[i + 2]
        valorDropout = configuracionConvRest[i + 3]
        model.add(Conv2D(numero_Filtros, (tamaño_Filtro, tamaño_Filtro), padding='same', strides=tamStrides, activation=funcion_Act))
        # Añado capa pooling
        if configuracionPooling.__contains__(indPooling):
            model.add(MaxPooling2D((2, 2)))
        indPooling += 1
        if valorBatch > 0:
            model.add(BatchNormalization())
        if valorDropout > 0:
            model.add(Dropout(valorDropout))
    numero_capasDensas = configuracion[2]

    print(f"Conf capas densas: {confCapasDensas}")
    model.add(Flatten())
    if numero_capasDensas > 0:
        # Para las capas densas
        for i in range(0,len(confCapasDensas),4):
            numFiltros = confCapasDensas[i]
            valorBatch = confCapasDensas[i+1]
            valorDropout = confCapasDensas[i+2]
            valorFuncActiv = confCapasDensas[i+3]
            model.add(Dense(numFiltros,activation=valorFuncActiv))
            if valorBatch > 0:
                model.add(BatchNormalization())
            if valorDropout > 0:
                model.add(Dropout(valorDropout))

    model.add(Dense(10, activation='softmax')) #Tengo que mantenerla porq necesito de salida 10 neuronas

    model.summary() #Esto muestra la red
    optimizador = configuracion[-2]

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizador,
                  metrics=['accuracy'])
    print("Training")
    model.fit(train_images, train_labels,steps_per_epoch=100, epochs=300, verbose=1, callbacks=[callback])

    print("Test")
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    return test_acc, test_loss


def cruce_CHC(padre1, padre2):
    #Solo cruzo la parte que me interesa
    hijo1 = np.array(padre1).copy()
    hijo2 = np.array(padre2).copy()

    #Me quedo con la parte que me interesa
    '''hijo1 = arrayLimpio(hijo1)
    hijo2 = arrayLimpio(hijo2)'''

    dif = np.abs(np.array(hijo1).copy(),np.array(hijo2).copy())
    dif[dif>0] = 1

    #La mitad de los 1 los borro
    diferenciasTotales = sum(dif)
    mitad = int(diferenciasTotales/2)

    cont = 0
    i = 0
    while cont < mitad:
        if dif[i] == 1:
            aleatorio = np.random.randint(0,2)
            if aleatorio > 0: #Si es mayor que 0, quito el 1
                dif[i] = 0
                cont += 1
        i += 1
        if i >= len(dif):
            i = 0

    cont = 0
    for ind in dif:
        cont+=1
        if dif[ind] > 0:
            valorP1 = hijo2[ind]
            valorP2 = hijo2[ind]

            hijo1[ind] = valorP2
            hijo2[ind] = valorP1

    '''hijo1 = rellenoConPadre(hijo1,padre1)
    hijo2 = rellenoConPadre(hijo2,padre2)'''

    print(f"Tam hijo1 antes de traduccion: {len(hijo1)}")
    print(f"Tam hijo2 antes de traduccion: {len(hijo2)}")
    hijo1 = gP.traduccionValida(hijo1)
    print(f"Tam hijo1 dsps de traduccion: {len(hijo1)}")
    hijo2 = gP.traduccionValida(hijo2)
    print(f"Tam hijo2 dsps de traduccion: {len(hijo2)}")

    '''#Mutacion hijo 1
    numAleatorio = np.random.rand(1)
    if numAleatorio < 0.03:
        posAleatoria = np.random.randint(0,len(hijo1))
        valorAleatorio = np.random.randint(1,8)
        hijo1[posAleatoria] = valorAleatorio
        hijo1 = gP.traduccionValida(hijo1)

    # Mutacion hijo 2
    numAleatorio = np.random.rand(1)
    if numAleatorio < 0.03:
        posAleatoria = np.random.randint(0, len(hijo2))
        valorAleatorio = np.random.randint(1, 8)
        hijo2[posAleatoria] = valorAleatorio
        hijo2 = gP.traduccionValida(hijo2)
    '''
    return hijo1,hijo2

def cruce_CHC_Alternativo(padre1, padre2):
    numCapasConvReal = v.getNumCapasConvReal()
    numCapasConv = v.getNumCapasConv()
    numCapasPooling = v.getNumCapasPooling()
    numCapasDensas = v.getNumCapasDensas()
    numCapasDensasReal = v.getNumCapasDensasReal()

    #Solo cruzo la parte que me interesa
    hijo1 = padre1.copy()
    hijo2 = padre2.copy()

    #Nuevos hijos
    hijo1Nuevo = []
    hijo2Nuevo = []
    #Cruzo numero capas
    print(f"\n---------------------------------"
          f"\n--------Entro en el cruce--------"
          f"\n---------------------------------")
    print(f"Padres : \n\tPadre1: {hijo1}\n\tPadre2: {hijo2}")
    print("Cruzo numero capas")
    capasHijo1 = np.array(hijo1[:3]).copy()
    capasHijo2 = np.array(hijo2[:3]).copy()
    capasHijo1,capasHijo2 = cruce(capasHijo1,capasHijo2)
    print(f"Capas cruzadas: \nC1: {capasHijo1}\nC2: {capasHijo2}")
    #Cambio valores hijo antes de limpiarlo
    for i in range(len(capasHijo1)):
        hijo1[i] = capasHijo1[i]
        hijo2[i] = capasHijo2[i]

    for e in capasHijo1:
        hijo1Nuevo.append(e)
    for e in capasHijo2:
        hijo2Nuevo.append(e)
    print(f"Hijos tras cruce capas: \n\tHijo1: {hijo1Nuevo}\n\tHijo2: {hijo2Nuevo}")
    #Numero capas
    capasConvH1 = hijo1Nuevo[0]
    capasConvH1Reales = numCapasConvReal[capasConvH1-1]
    capasPoolH1 = hijo1Nuevo[1]
    capasDensH1 = hijo1Nuevo[2]
    capasDensH1Reales = numCapasDensasReal[capasDensH1 - 1]

    capasConvH2 = hijo2Nuevo[0]
    capasConvH2Reales = numCapasConvReal[capasConvH2 - 1]
    capasPoolH2 = hijo2Nuevo[1]
    capasDensH2 = hijo2Nuevo[2]
    capasDensH2Reales = numCapasDensasReal[capasDensH2 - 1]

    #Valores inicio
    inicioPoolingH1 = (inicioConv) + (capasConvH1Reales * 4)
    inicioDensasH1 = inicioPoolingH1 + capasPoolH1

    inicioPoolingH2 = (inicioConv) + (capasConvH2Reales * 4)
    inicioDensasH2 = inicioPoolingH2 + capasPoolH2

    #-------------
    #Cruzo conv
    # -------------
    print("----------Conv---------")
    if capasConvH1Reales > capasConvH2Reales: #El maximo lo marca la mas grande
        print("Caso 1")
        convHijo1 = hijo1[inicioConv:inicioConv + (capasConvH1Reales * 4)].copy()
        convHijo2 = hijo2[inicioConv:inicioConv + (capasConvH1Reales * 4)].copy()
    elif capasConvH2Reales > capasConvH1Reales: #El maximo lo marca la mas grande
        print("Caso 2")
        convHijo1 = hijo1[inicioConv:inicioConv + (capasConvH2Reales * 4)].copy()
        convHijo2 = hijo2[inicioConv:inicioConv + (capasConvH2Reales * 4)].copy()
    else:
        print("Caso 3")
        convHijo1 = hijo1[inicioConv:inicioConv+ (capasConvH1Reales*4) ].copy()
        convHijo2 = hijo2[inicioConv:inicioConv+ (capasConvH2Reales*4) ].copy()
    #Los cruzo
    print(f"Cruzo conv: \n\tConv1: {convHijo1}\n\tConv2: {convHijo2}")
    convHijo1, convHijo2 = cruce(convHijo1, convHijo2)

    if capasConvH1Reales > capasConvH2Reales: #Si el 1 es mas grande
        convHijo2 = convHijo2[:capasConvH2Reales*4]
    elif capasConvH2Reales > capasConvH1Reales: #Si el 2 es mas grande
        convHijo1 = convHijo1[:capasConvH1Reales*4]

    restanteConvHijo1 = hijo1[inicioConv + (capasConvH1Reales * 4):inicioPooling].copy()
    restanteConvHijo2 = hijo2[inicioConv + (capasConvH2Reales * 4):inicioPooling].copy()
    for e in convHijo1:
        hijo1Nuevo.append(e)
    for e in restanteConvHijo1:
        hijo1Nuevo.append(e)
    for e in convHijo2:
        hijo2Nuevo.append(e)
    for e in restanteConvHijo2:
        hijo2Nuevo.append(e)

    tamHastaPoolingH1Nuevo = len(hijo1Nuevo)
    tamHastaPoolingPadre1 = len(hijo1[:inicioPooling])
    print(f"Tamaños: \n\tHijo nuevo 1: {tamHastaPoolingH1Nuevo}\n\tPadre 1: {tamHastaPoolingPadre1}")
    print(f"Hijos tras cruce conv: \n\tHijo1: {hijo1Nuevo}\n\tHijo2: {hijo2Nuevo}")
    print(f"Padres tras cruce conv: \n\tPadre1: {hijo1}\n\tPadre2: {hijo2}")
    # -------------
    #Cruzo pooling
    # -------------
    print("----------Pooling----------")
    if capasPoolH1 > capasPoolH2:
        poolHijo1 = hijo1[inicioPooling: inicioPooling + capasPoolH1]
        poolHijo2 = hijo2[inicioPooling: inicioPooling + capasPoolH1]
    elif capasPoolH2 > capasPoolH1:
        poolHijo1 = hijo1[inicioPooling: inicioPooling + capasPoolH2]
        poolHijo2 = hijo2[inicioPooling: inicioPooling + capasPoolH2]
    else:
        poolHijo1 = hijo1[inicioPooling: inicioPooling + capasPoolH1]
        poolHijo2 = hijo2[inicioPooling: inicioPooling + capasPoolH2]

    #Los cruzo
    print(f"Cruzo pooling: \n\tPooling1: {poolHijo1}\n\tPooling2: {poolHijo2}")
    poolHijo1, poolHijo2 = cruce(poolHijo1, poolHijo2)
    if capasPoolH1 > capasPoolH2:
        poolHijo2 = poolHijo2[:capasPoolH2]
    elif capasPoolH2 > capasPoolH1:
        poolHijo1 = poolHijo1[:capasPoolH1]

    restantePoolHijo1 = hijo1[inicioPooling + capasPoolH1: inicioDensas].copy()
    restantePoolHijo2 = hijo2[inicioPooling + capasPoolH2: inicioDensas].copy()


    for e in poolHijo1:
        hijo1Nuevo.append(e)
    for e in restantePoolHijo1:
        hijo1Nuevo.append(e)
    for e in poolHijo2:
        hijo2Nuevo.append(e)
    for e in restantePoolHijo2:
        hijo2Nuevo.append(e)
    tamHastaDensasH1Nuevo = len(hijo1Nuevo)
    tamHastaDensasPadre1 = len(hijo1[:inicioDensas])
    print(f"Tamaños: \n\tHijo nuevo 1: {tamHastaDensasH1Nuevo}\n\tPadre 1: {tamHastaDensasPadre1}")
    print(f"Hijos tras cruce pooling: \n\tHijo1: {hijo1Nuevo}\n\tHijo2: {hijo2Nuevo}")
    print(f"Padres tras cruce pooling: \n\tPadre1: {hijo1}\n\tPadre2: {hijo2}")
    # -------------
    #Cruzo densas
    # -------------
    print("----------Densas----------")
    if capasDensH1Reales > capasDensH2Reales:
        print("Caso 1")
        densasHijo1 = hijo1[inicioDensas: inicioDensas + (capasDensH1Reales*4) ].copy()
        densasHijo2 = hijo2[inicioDensas: inicioDensas + (capasDensH1Reales*4)].copy()
    elif capasDensH2Reales > capasDensH1Reales:
        print("Caso 2")
        densasHijo1 = hijo1[inicioDensas: inicioDensas + (capasDensH2Reales*4) ].copy()
        densasHijo2 = hijo2[inicioDensas: inicioDensas + (capasDensH2Reales*4)].copy()
    else:
        print("Caso 3")
        densasHijo1 = hijo1[inicioDensas: inicioDensas + (capasDensH1Reales * 4)].copy()
        densasHijo2 = hijo2[inicioDensas: inicioDensas + (capasDensH2Reales * 4)].copy()

    #Los cruzo
    print(f"Cruzo densas: \n\tDensas1: {densasHijo1}\n\tDensas2: {densasHijo2}")
    densasHijo1, densasHijo2 = cruce(densasHijo1, densasHijo2)
    if capasDensH1Reales > capasDensH2Reales:
        densasHijo2 = densasHijo2[:capasDensH2Reales*4]
    elif capasDensH2Reales > capasDensH1Reales:
        densasHijo1 = densasHijo1[:capasDensH1Reales*4]

    print(f"Densas tras cruce: \n\tDensas1: {densasHijo1}\n\tDensas2: {densasHijo2}")
    restanteDensasHijo1 = hijo1[inicioDensas + (capasDensH1Reales*4): -3]
    restanteDensasHijo2 = hijo2[inicioDensas + (capasDensH2Reales*4): -3]
    print(f"Restante densas: \n\trestanteDensasHijo1: {restanteDensasHijo1}\n\trestanteDensasHijo2: {restanteDensasHijo2}")
    for e in densasHijo1:
        hijo1Nuevo.append(e)
    for e in restanteDensasHijo1:
        hijo1Nuevo.append(e)
    for e in densasHijo2:
        hijo2Nuevo.append(e)
    for e in restanteDensasHijo2:
        hijo2Nuevo.append(e)
    tamHastaFinalH1Nuevo = len(hijo1Nuevo)
    tamHastaFinalPadre1 = len(hijo1[:-3])
    print(f"Tamaños: \n\tHijo nuevo 1: {tamHastaFinalH1Nuevo}\n\tPadre 1: {tamHastaFinalPadre1}")
    print(f"Hijos tras cruce densas: \n\tHijo1: {hijo1Nuevo}\n\tHijo2: {hijo2Nuevo}")
    print(f"Padres tras cruce densas: \n\tPadre1: {hijo1}\n\tPadre2: {hijo2}")
    #------------------------
    #Cruzo stride,opt y fact
    #------------------------
    print("-----------Final--------------")
    ultimosValoresH1 = hijo1[-3:]
    ultimosValoresH2 = hijo2[-3:]
    print("Cruzo stride y cosas")
    ultimosValoresH1,ultimosValoresH2 = cruce(ultimosValoresH1,ultimosValoresH2)

    for e in ultimosValoresH1:
        hijo1Nuevo.append(e)
    for e in ultimosValoresH2:
        hijo2Nuevo.append(e)


    print(f"Tam hijo1 antes de traduccion: {len(hijo1Nuevo)}")
    print(f"Tam hijo2 antes de traduccion: {len(hijo2Nuevo)}")
    hijo1 = gP.traduccionValida(hijo1Nuevo)
    print(f"Tam hijo1 dsps de traduccion: {len(hijo1)}")
    hijo2 = gP.traduccionValida(hijo2Nuevo)
    print(f"Tam hijo2 dsps de traduccion: {len(hijo2)}")

    if len(hijo1) != len(padre1):
        raise Exception("Cruce mal hecho, padre con distinto tamaño que el hijo")
    return hijo1,hijo2


def torneo(puntuacion, k):
    candidatos = list()
    indices = list()
    for i in range(k):
        n = int(np.random.uniform(0, len(puntuacion)))
        while indices.__contains__(n):
            n += 1
            if n >= len(puntuacion):
                n = 0
        indices.append(n)
        candidatos.append(puntuacion[n])

    # print(f"Candidatos: {candidatos}")
    candidatos = np.array(candidatos)
    maximoIndice = np.argmax(candidatos)
    padre = indices[maximoIndice]

    # Deberia de seleccionar el mejor padre
    return padre


def reemplazo(poblacion, puntuacion, hijos, puntosHijos):  # Reemplazo a los elementos que sean peores que los hijos
    indiceMenor = np.argmin(puntuacion)  # Quito el que menos puntos tenga
    hijoMayor = np.argmax(puntosHijos)
    # Sustituyo un hijo
    if puntuacion[indiceMenor] < puntosHijos[hijoMayor] or math.isnan(puntuacion[indiceMenor]):
        puntuacion[indiceMenor] = puntosHijos[hijoMayor]
        poblacion[indiceMenor] = hijos[hijoMayor]
    # Sustituyo el otro hijo
    indiceMenor = np.argmin(puntuacion)
    hijoMenor = np.argmin(puntosHijos)

    if puntuacion[indiceMenor] < puntosHijos[hijoMenor] or math.isnan(puntuacion[indiceMenor]):
        puntuacion[indiceMenor] = puntosHijos[hijoMenor]
        poblacion[indiceMenor] = hijos[hijoMenor]
    return poblacion, puntuacion


def selecciono_Mejores(poblaciones, puntuacion, cuantos):
    print(f"Al quedarme con los mejores tengo")
    mejorPoblacion = []
    mejorPuntuacion = []

    poblacion = np.array(poblaciones).copy()
    puntuacion = list(np.array(puntuacion))
    indicesMejores = (np.flip(np.argsort(puntuacion))[0:cuantos])

    for indice in indicesMejores:
        mejorPoblacion.append(list(poblacion[indice]))
        mejorPuntuacion.append(puntuacion[indice])


    return mejorPoblacion, mejorPuntuacion



def traducirConfiguracion(configuracion):
    #Tengo que traducir toda la configuracion completa, para que los valores de inicio de vectores se mantengan
    numCapasConvReal = v.getNumCapasConvReal()
    numCapasConv = v.getNumCapasConv()
    numCapasPooling = v.getNumCapasPooling()
    numCapasDensas = v.getNumCapasDensas()
    numCapasDensasReal = v.getNumCapasDensasReal()
    tamFiltros = v.getTamFiltrosReal()
    numFiltros = v.getNumFiltrosReal()
    numFiltrosDensas = v.getNumFiltrosReal()
    stride = v.getStride()  # Pasos en que se mueve la ventana de los filtros
    padding = v.getPaddingReal()
    Factivacion = v.getFactivacionReal()
    optimizadores = v.getOptimizadoresReal()
    dropout = v.getDropoutReal()
    batchNorm = v.getBatchNormReal()


    configuracionTraducida = []

    capasConf = configuracion[0]
    capasPooling = configuracion[1]
    capasDensas = configuracion[2]
    # Añado el numero de capas traducido
    numero_capasConv = numCapasConvReal[capasConf - 1]
    configuracionTraducida.append(numero_capasConv)
    # Añado el numero de capas pooling
    numCapasPoolingReal = numCapasPooling[capasPooling - 1]
    configuracionTraducida.append(numCapasPoolingReal)
    #Añado el numero de capas densas
    numero_capasD = numCapasDensas[capasDensas - 1]
    configuracionTraducida.append(numero_capasD)

    print(f"inicio conv: {inicioConv}")
    print(f"inicio pooling: {inicioPooling}")
    print(f"inicio densas: {inicioDensas}")
    confConv = configuracion[inicioConv:inicioPooling]
    print(f"Conf restante conv: {confConv}")
    #Añado valores para cada capa conv
    for i in range(0, len(confConv), 4):
        numFiltrosConf = numFiltros[confConv[i] - 1]
        tamFiltroConf = tamFiltros[confConv[i+1] - 1]
        valorBatch = batchNorm[confConv[i+2] - 1]
        valorDropout = dropout[confConv[i+3] - 1]

        configuracionTraducida.append(numFiltrosConf)
        configuracionTraducida.append(tamFiltroConf)
        configuracionTraducida.append(valorBatch)
        configuracionTraducida.append(valorDropout)

    # Obtengo vector posiciones pooling
    vectorPosicionesPooling = np.array(configuracion[inicioPooling:inicioDensas]).copy()
    # Añado vector de posiciones de pooling
    for valor in vectorPosicionesPooling:
        configuracionTraducida.append(valor)
    #Para las capas densas
    confCapasDensas = configuracion[inicioDensas:-3]
    print(f"Conf capas densas: {confCapasDensas}")

    for i in range(0,len(confCapasDensas),4):
        numFiltrosConf = numFiltrosDensas[confCapasDensas[i] - 1]
        valorBatch = batchNorm[confCapasDensas[i + 1] - 1]
        valorDropout = dropout[confCapasDensas[i + 2] - 1]
        valorFuncActiv = Factivacion[confCapasDensas[i + 3] - 1]
        configuracionTraducida.append(numFiltrosConf)
        configuracionTraducida.append(valorBatch)
        configuracionTraducida.append(valorDropout)
        configuracionTraducida.append(valorFuncActiv)

    # Stride
    strideConf = configuracion[-3]
    configuracionTraducida.append(stride[strideConf - 1])
    #Optimizador
    optimConf = configuracion[-2]
    configuracionTraducida.append(optimizadores[optimConf - 1])
    #F activ
    fActConf = configuracion[-1]
    configuracionTraducida.append(Factivacion[fActConf - 1])

    return configuracionTraducida


def puntuacionCromosoma(individuo, imagenesPasadas):
    try:
        acc, loss = redNeuronal_CNN(individuo, imagenes=imagenesPasadas)
    except:
        acc = 0
    #acc = fEvaluacion_Alternativa(individuo,imagenesPasadas)
    return acc

def fEvaluacion_Alternativa(individuo,tabla):
    numeroFilas = 50
    sumaTotal = 0
    print(len(individuo))
    for i in range(numeroFilas):
        sumFila = sum(individuo[j]*tabla[i][j] for j in range(len(individuo)))
        sumaTotal += sumFila
    return sumaTotal

def generaTabla():
    tabla = list()
    numeroFilas = 50
    for i in range(numeroFilas):
        fila = random.sample(range(-200,300),90)
        tabla.append(fila)
    return tabla

def cruce(hijo1, hijo2):
    #Tienen que medir lo mismo ambos hijos
    dif = np.abs(np.array(hijo1).copy(), np.array(hijo2).copy())
    dif[dif > 0] = 1

    # La mitad de los 1 los borro
    diferenciasTotales = sum(dif)
    mitad = int(diferenciasTotales / 2)

    cont = 0
    i = 0
    while cont < mitad:
        if dif[i] == 1:
            aleatorio = np.random.randint(0, 2)
            if aleatorio > 0:  # Si es mayor que 0, quito el 1
                dif[i] = 0
                cont += 1
        i += 1
        if i >= len(dif):
            i = 0

    cont = 0
    for ind in dif:

        if ind > 0:
            valorP1 = hijo1[cont]
            valorP2 = hijo2[cont]

            hijo1[cont] = valorP2
            hijo2[cont] = valorP1
        cont += 1

    return hijo1, hijo2


def añadeVectPosiciones(cromosoma, vectorPosiciones):
    print(f"Añade a {cromosoma} el vector {vectorPosiciones}")
    cromosoma = list(cromosoma)
    padreAux = []
    padreAux.append(cromosoma[0])
    padreAux.append(vectorPosiciones)
    padreAux = padreAux + cromosoma[1:]

    return padreAux


def reemplazoWAMS(poblacion, puntuacion, hijos, puntosHijos):  # No necesito los puntos de los hijos
    for i in range(len(hijos)):
        hijo = hijos[i]
        puntosHijo = puntosHijos[i]

        puntos = []
        for individuo in poblacion:
            puntos.append(diferencia(hijo, individuo))
        indices = np.argsort(puntos)[0:3]  # Indices de los padres que mas se parecen
        puntuacionesTorneo = [puntuacion[indices[0]], puntuacion[indices[1]],
                              puntuacion[2]]  # Puntuaciones de los padres que mas se parecen
        peorPadreTorneo = indices[np.argmin(puntuacionesTorneo)]

        # Solo lo cambio si mejora
        if puntosHijo > puntuacion[peorPadreTorneo]:
            poblacion[peorPadreTorneo] = hijo
            puntuacion[peorPadreTorneo] = puntosHijo

    return poblacion, puntuacion


def diferencia(hijo, padre):
    hijo = np.array(hijo.copy())
    padre = np.array(padre.copy())

    if len(hijo) > len(padre):
        aux = np.zeros(len(hijo))
        aux[0:len(padre)] = padre
        padre = aux
    elif len(padre) > len(hijo):
        aux = np.zeros(len(padre))
        aux[0:len(hijo)] = hijo
        hijo = aux

    return np.sum(np.abs(hijo - padre)).__int__()


def reinicializo(poblacion, puntuacion, tamElite):
    poblacionNueva = []
    puntuacionNueva = []
    indices = np.flip(np.argsort(puntuacion))[0:tamElite]  # Me quedo con los "tamElite" mejores
    for indice in indices:
        poblacionNueva.append(poblacion[indice])
        puntuacionNueva.append(puntuacion[indice])

    return poblacionNueva, puntuacionNueva


def pintarGrafica(x,y,pinta,titulo):

    print(x)
    print(y)
    y = np.array(y)
    y = np.round(y, 6)
    maximoValor = y[y.argmax()]
    minimoValor = y[y.argmin()]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(minimoValor , maximoValor)
    plt.plot(x, y)

    for i, j in zip(x, y):
        ax.annotate(str(j), xy=(i, j))
    if pinta:
        plt.title(titulo)
        plt.show()
    print("Termino de pintar")

def printPoblacion_Puntuacion(mensaje, poblacion, puntuacion):
    print(f"\n\t\t--------Poblacion{mensaje}: ---------")
    for k in range(len(poblacion)):
        print(f"Individuo {k}: {puntuacion[k]} || {poblacion[k]} ")

def escribeFichero(dirFichero, listaEv, listaPun,reinicios,mejorCromosoma):
    print("Escribo:")
    print(f"Mejor cromosoma: {mejorCromosoma}")
    fichero = open(dirFichero, 'w')
    listas = [listaEv, listaPun,reinicios,mejorCromosoma]

    for l in listas:
        fichero.write(l.__str__() + '\n')
    fichero.close()


