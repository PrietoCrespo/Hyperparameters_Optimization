import numpy as np
import math
import funcionesGenetico as fG
import variables as v

inicioConv = 3
inicioPooling = inicioConv + 4 * v.getNumCapasConvReal()[-1]
inicioDensas = inicioPooling + v.getNumCapasPooling()[-1]


def generaIndividuo():
    print("Genero individuo")
    numCapasConvReal = v.getNumCapasConvReal()
    numCapasConv = v.getNumCapasConv()
    numCapasPooling = v.getNumCapasPooling()
    numCapasDensasReal = v.getNumCapasDensasReal()
    numCapasDensas = v.getNumCapasDensas()
    tamFiltros = v.getTamFiltros()
    numFiltros = v.getNumFiltros()
    stride = v.getStride()  # Lo sustituyo por los optimizadores
    padding = v.getPadding()
    Factivacion = v.getFactivacion()
    optimizadores = v.getOptimizadores()
    batchNorm = v.getBatchNormReal()
    dropout = v.getDropout()

    inicioPooling = inicioConv + 4 * numCapasConvReal[-1]
    inicioDensas = inicioPooling + numCapasPooling[-1]
    # Añado num capas
    config = []
    # Obtengo el numero de capas conv. que voy a tener
    capasConv = np.random.randint(1, len(numCapasConv) + 1)
    # Obtengo el numero de capas pooling que voy a tener
    capasPooling = np.random.randint(1, len(numCapasPooling) + 1)
    # El numero de capas de pooling tiene que ser menor o igual que el de convolucionales, me aseguro así
    while capasPooling > numCapasConvReal[capasConv - 1]:
        capasPooling = np.random.randint(1, len(numCapasPooling) + 1)
    # Obtengo el numero de capas densas
    capasDens = np.random.randint(1, numCapasDensas[-1] + 1)
    # Añado el numero de capas conv
    config.append(capasConv)
    # Añado el numero de capas pooling
    config.append(capasPooling)
    # Añado el numero de capas densas
    config.append(capasDens)
    vPosiciones = []
    # Añado las capas convolucionales
    for j in range(numCapasConvReal[-1]):
        # Por cada capa
        # Numero  filtros
        nfcc = np.random.randint(1, numFiltros[-1] + 1)
        config.append(nfcc)
        # Tam filtros
        tfcc = np.random.randint(1, tamFiltros[-1] + 1)
        config.append(tfcc)
        # Batch normalization
        bN = np.random.randint(1, batchNorm[-1] + 1)
        config.append(bN)
        # dropout
        dO = np.random.randint(1, dropout[-1] + 1)
        config.append(dO)
    # Añado las capas de pooling
    for i in range(capasPooling):
        pos = np.random.randint(1, numCapasConvReal[capasConv - 1] + 1)
        print(vPosiciones)
        while vPosiciones.__contains__(pos):  # Evita repetidos
            pos = np.random.randint(1, numCapasConvReal[capasConv - 1] + 1)
            print(pos)
        config.append(pos)
        vPosiciones.append(pos)

    # Una vez obtengo los valores pooling buenos, relleno el vector con valores aleatorios sin que se repitan
    for z in range(numCapasPooling[-1] - capasPooling):
        pos = np.random.randint(1, numCapasConvReal[-1] + 1)
        print(vPosiciones)
        while vPosiciones.__contains__(pos):  # Evita repetidos
            pos = np.random.randint(1, numCapasConvReal[-1] + 1)
            print(pos)
        config.append(pos)
        vPosiciones.append(pos)

    # Añado las capas densas
    for k in range(numCapasDensasReal[-1]):
        # Numero  neuronas
        nfcc = np.random.randint(1, numFiltros[-1] + 1)
        config.append(nfcc)
        # Batch normalization
        bN = np.random.randint(1, batchNorm[-1] + 1)
        config.append(bN)
        # dropout
        dO = np.random.randint(1, dropout[-1] + 1)
        config.append(dO)
        # F activ
        funcActiv = np.random.randint(1, len(Factivacion) + 1)
        config.append(funcActiv)

    # Stride
    pasos = np.random.randint(1, stride[-1] + 1)
    config.append(pasos)
    # Optimizadores
    optimizador = np.random.randint(1, optimizadores[-1] + 1)
    config.append(optimizador)
    # FActiv
    funcion = np.random.randint(1, len(Factivacion) + 1)
    config.append(funcion)
    print(f"Individuo generado: {config}")
    print(f"Tamaño individuo: {len(config)}")
    return config


def generaPoblacion(numeroPoblacion, baseDatos):
    if baseDatos <= 1:
        poblacion, puntuacion, evaluaciones = generaPoblacion_Ropa_Numeros(nPoblacion=numeroPoblacion,
                                                                           baseDatos=baseDatos)

    return poblacion, puntuacion, evaluaciones


def generaPoblacion(nPoblacion, baseDatos, imagenes):
    # Traduccion

    evaluaciones = 0
    poblacion = []
    puntuacion = []
    for i in range(nPoblacion):
        # Genero el individuo
        config = generaIndividuo()
        # Evaluo el individuo
        print(f"Individuo numero: {i}")
        score = fG.puntuacionCromosoma(config, baseDatos=baseDatos, imagenesPasadas=imagenes)
        if score > 0:
            evaluaciones += 1
        # Busco individuos hasta que uno sea valido
        # Individuos validos: aquellos cuya precision sea menor que 0.3
        buscados = 0
        while score < 0.3 and buscados < 4:
            print("\tBusco a otro")
            config = generaIndividuo()
            score = fG.puntuacionCromosoma(config, baseDatos=baseDatos, imagenesPasadas=imagenes)
            buscados += 1
            if score > 0:
                evaluaciones += 1
        poblacion.append(config)
        puntuacion.append(score)
    return poblacion, puntuacion, evaluaciones


def traduccionValida(configuracion):
    # Comprueba si la configuracion es valida, y en caso de que no sea valida la devuelve corregida
    # Primero compruebo que las capas pooling estén bien, tanto el número que hay como el máximo
    # Segundo compruebo que las capas convolucionales están bien; voy recorriendo cada una de las capas y compruebo que todos sus valores están bien, así como que el número de capas indicado es el correcto
    # Tercero compruebo que el número de capas densas están bien, tanto el número como sus valores
    # Comprobar que aun me quedan 3 valores más por cubrir
    '''
    numCapasConvReal = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    numCapasConv = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    numCapasDensas = [1, 2, 3, 4, 5, 6]
    numCapasPooling = [1, 2, 3, 4, 5]
    tamFiltros = [3, 5, 7]
    numFiltros = [32, 64, 128, 256, 512]
    numFiltrosDensas = [10,32, 64, 128, 256]
    stride = [1, 2]  # Pasos en que se mueve la ventana de los filtros
    padding = ['valid', 'same']
    Factivacion = ['elu', 'exponential', 'tanh', 'sigmoid', 'softmax', 'relu', 'selu']
    optimizadores = ['SGD','RMSprop','Adam']
    dropout = [0, 0.25, 0.5]
    batchNorm = [0, 1]
    '''
    numCapasConvReal = v.getNumCapasConvReal()
    numCapasConv = v.getNumCapasConv()
    numCapasPooling = v.getNumCapasPooling()
    valoresCapasDensas = v.getNumCapasDensas()
    numCapasDensas = len(v.getNumCapasDensas())
    numFiltros = len(v.getNumFiltros())
    tamFiltros = len(v.getTamFiltros())
    stride = len(v.getStride())
    padding = len(v.getPadding())
    optimizadores = len(v.getOptimizadores())
    Factivacion = len(v.getFactivacion())
    valoresBatch = len(v.getBatchNorm())
    valoresDropout = len(v.getDropout())

    tamConf = len(configuracion)
    numeroParametrosAparte = 6

    configuracionDevuelta = []

    print("Traduccion valida")

    capasConv = reajuste(configuracion[0], numCapasConv[-1])
    print(f"Capas conv: {capasConv}")
    configuracionDevuelta.append(capasConv)
    capasPooling = reajuste(configuracion[1], numCapasConvReal[capasConv - 1])
    print(f"Capas pooling: {capasPooling}")
    configuracionDevuelta.append(capasPooling)
    capasDensas = reajuste(configuracion[3], valoresCapasDensas[-1])
    configuracionDevuelta.append(capasDensas)
    # Obtengo los distintos vectores
    vectorCapasConv = np.array(configuracion[inicioConv:inicioPooling]).copy()
    for valor in vectorCapasConv:
        configuracionDevuelta.append(valor)
    vectorCapasPooling = np.array(configuracion[inicioPooling:inicioPooling + numCapasPooling[-1]]).copy()
    vectorCapasDensas = np.array(configuracion[inicioDensas:-3]).copy()
    # En principio solo tengo que reajustar el vector de capas pooling
    # Reajusto pooling
    # Primero obtengo la parte que me interesa
    poolingPeque = vectorCapasPooling[:capasPooling]
    print(f"Pooling peque: {poolingPeque}")
    poolingReajustado = []
    valoresPosibles = []
    print("Aqui")
    for valor in poolingPeque:
        valor = reajuste(valor, numCapasConvReal[capasConv - 1])
        print(f"\tPooling reajustado: {poolingReajustado}")
        print(f"\tValor: {valor}")
        while poolingReajustado.__contains__(valor):
            valor = np.random.randint(1, numCapasConvReal[capasConv - 1] + 1)
            # print(f"\t\tPruebo otro valor: {valor}")
        poolingReajustado.append(valor)
        configuracionDevuelta.append(valor)
    # Ahora corrijo la parte que menos me interesa
    poolingResto = vectorCapasPooling[capasPooling:]
    print(f"Pooling resto: {poolingResto}")
    print("Me quedo aqui")
    for valor in poolingResto:
        valor = reajuste(valor, numCapasConvReal[-1])
        while poolingReajustado.__contains__(valor):
            valor = np.random.randint(1, numCapasConvReal[-1] + 1)
        poolingReajustado.append(valor)
        configuracionDevuelta.append(valor)
    print("O aqui")
    for valor in vectorCapasDensas:
        configuracionDevuelta.append(valor)
    valorStride = configuracion[-3]
    valorOpt = configuracion[-2]
    valorFact = configuracion[-1]
    configuracionDevuelta.append(valorStride)
    configuracionDevuelta.append(valorOpt)
    configuracionDevuelta.append(valorFact)

    configuracionDevuelta = list(configuracionDevuelta)

    return configuracionDevuelta


def corregirCapas(configuracion):
    # Considero que el cruce no modifica el tamaño de los hijos (son del mismo tamaño que los padres)
    # Le estoy dando prioridad al numero de capas que tengo (elementos en el array) sobre el numero de capas indicado
    numCapasConv = v.getNumCapasConv()
    numeroCapas = len(configuracion[1:-3]) / 2
    print(f"Tiene {numeroCapas} capas")
    if numeroCapas % 1 != 0:
        raise Exception("Red neuronal mal construida (Número de capas decimal)")
    valorCorregido = ((len(configuracion[1:-3]) / 2) - 2).__int__()
    print(f"Corrijo y le pongo de valor: {valorCorregido}")
    configuracion[0] = valorCorregido

    return configuracion


def reajuste(valorRecibido, maximoPosible):  # Reajuste de los valores

    if valorRecibido > maximoPosible:
        modulo = valorRecibido % maximoPosible
        valorRecibido = maximoPosible - modulo
    return valorRecibido


'''conf = traduccionValida([2, 1, 2, 2, 2, 1, 2, 3, 3])
print(conf)'''

if __name__ == "__main__":
    poblacion, puntuacion = generaPoblacionRopa(5)

    print("\n\t\t--------Poblacion: ---------")
    for k in range(len(poblacion)):
        print(f"Individuo {k}: {puntuacion[k]} || {poblacion[k]} ")
