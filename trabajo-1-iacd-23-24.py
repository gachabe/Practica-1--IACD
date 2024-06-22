# Inteligencia Artificial para la Ciencia de los Datos
# Implementación de clasificadores 
# Dpto. de C. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS:Chaves Benitez
# NOMBRE: Gabriel
#
# Segundo componente (si se trata de un grupo):
#
# APELLIDOS: Sanchez Trigo
# NOMBRE: Francisco Horacio
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo
# que debe realizarse de manera individual o con la pareja del grupo. 
# La discusión y el intercambio de información de carácter general con los 
# compañeros se permite, pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir 
# código de terceros, OBTENIDO A TRAVÉS DE LA RED, 
# DE HERRAMIENTAS DE GENERACIÓN DE CÓDIGO o cualquier otro medio, 
# se considerará plagio. Si tienen dificultades para realizar el ejercicio, 
# consulten con el profesor. En caso de detectarse plagio, supondrá 
# una calificación de cero en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************


# MUY IMPORTANTE: 
# ===============    
    
# * NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
#   Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# * En este trabajo NO SE PERMITE USAR Scikit Learn. 
  
# * Se recomienda (y se valora especialmente) el uso eficiente de numpy. Todos 
#   los datasets se suponen dados como arrays de numpy. 

# * Este archivo (con las implementaciones realizadas), ES LO ÚNICO QUE HAY QUE ENTREGAR.
#   AL FIONAL DE ESTE ARCHIVO hay una serie de ejemplos a ejecutar que están comentados, y que
#   será lo que se ejecute durante la presentación del trabajo al profesor.
#   En la versión final a entregar, descomentar esos ejemplos del final y no dejar 
#   ninguna otra ejecución de ejemplos. 



import math
import random
import numpy as np



# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar (casi) todos los conjuntos de datos,
# basta con tener descomprimido el archivo datos-trabajo-1-iacd.tgz (en el mismo sitio
# que este archivo) Y CARGARLOS CON LA SIGUIENTE ORDEN. 
    
from carga_datos import *    

# Como consecuencia de la línea anterior, se habrán cargado los siguientes 
# conjuntos de datos, que pasamos a describir, junto con los nombres de las 
# variables donde se cargan. Todos son arrays de numpy: 


# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (0:republicano o 1:demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   


# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.

  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos del dataset original. 
#   Los textos se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# Además, en la carpeta datos/digitdata se tiene el siguiente dataset, que
# habrá de ser procesado y cargado:  

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En la carpeta digitdata están todos los datos.
#   Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 



# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#particion_entr_prueba(X,y,test=0.20)
def particion_entr_prueba(X,y,test=0.20):
    if 0 < test or test < 1:
        # Obtenemos el número total de muestras
        n_muestras = len(y)
        n_muestras_train = int(n_muestras * (1 - test))
        valores_dif = np.unique(y, return_counts=True) # Cuales son los valores de clasificacion que hay y cuantos de cada
        indices_train = np.array([], dtype=int) # Sin dtype no funciona el concatenate, supongo que como no conoce el tipo de dato que v a intodcir no lo hace
        indices_test = np.array([], dtype=int)
        for valor, cantidad in zip(*valores_dif):
            indice_valor = np.where(y == valor)[0] # Encontramos las posiciones para el valor dado
            proporcion = int((cantidad/n_muestras)*n_muestras_train)
            random.shuffle(indice_valor) # Barajo los indices
            indices_train_valor = indice_valor[:proporcion] # cojo los n primero indices
            indices_test_valor = indice_valor[proporcion:]
            indices_test = np.concatenate((indices_test, indices_test_valor)) #añado a la lista de indices lo indices obtenido para el valor de clasificacion anterior
            indices_train = np.concatenate((indices_train, indices_train_valor))
        indices_train.sort() # Ordeno para mantener la clasificacion
        indices_test.sort()

        return X[indices_train], X[indices_test], y[indices_train], y[indices_test]
    else:
        raise Exception("El valor de proporcionalidad debe estar entre 0 y 1. ")

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, el orden
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   
# 

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

#Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
#print(y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0])
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

#np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
#print(np.unique(ye_votos,return_counts=True))
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------
























# ========================================================
# EJERCICIO 2: IMPLEMENTACIÓN DEL CLASIFICADOR NAIVE BAYES
# ========================================================

# Se pide implementar el clasificador Naive Bayes, en su versión categórica
# con suavizado y log probabilidades (descrito en el tema 2, diapositivas 22 a
# 34). En concreto:


# ----------------------------------
# 2.1) Implementación de Naive Bayes
# ----------------------------------

# Definir una clase NaiveBayes con la siguiente estructura:

# class NaiveBayes():

#     def __init__(self,k=1):
#                 
#          .....
         
#     def entrena(self,X,y):

#         ......

#     def clasifica_prob(self,ejemplo):

#         ......

#     def clasifica(self,ejemplo):

#         ......


# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1) 
# * Método entrena, recibe como argumentos dos arrays de numpy, X e y, con los
#   datos y los valores de clasificación respectivamente. Tiene como efecto el
#   entrenamiento del modelo sobre los datos que se proporcionan. NOTA: Se valorará
#   que el entrenamiento se haga con un único recorrido del dataset. 
# * Método clasifica_prob: recibe un ejemplo (en forma de array de numpy) y
#   devuelve una distribución de probabilidades (en forma de diccionario) que
#   a cada clase le asigna la probabilidad que el modelo predice de que el
#   ejemplo pertenezca a esa clase. 
# * Método clasifica: recibe un ejemplo (en forma de array de numpy) y
#   devuelve la clase que el modelo predice para ese ejemplo.   

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception):
    "Error, clasificador aun no entrenado"
    pass

class NaiveBayes():

    def __init__(self, k=1):
        self.k = k
        self.clases = None
        self.n_clases = None
        self.n_atrib = None
        self.prob_clase_atrib = None

    def entrena(self, X, y):
        # Iniciamos los atributos de la clase
        self.clases, cant_elem_clase = np.unique(y, return_counts=True)
        self.n_clases = len(self.clases)
        self.n_atrib = X.shape[1]
        self.prob_clases = cant_elem_clase/len(y)
        # Guardaremos las probabilidades condicionadas en un diccionario para mayor comodidad
        self.prob_clase_atrib = {}
        # Calculamos la cantidad de valores diferentes que toma cada atributo para aplicar suavizado
        self.cantidad_valores_atrib = np.apply_along_axis(lambda col: len(np.unique(col)), axis=0, arr=X)
        # Damos una lista de lista donde cada sublista son los valores posibles por atributo
        self.valores_atrib= [np.unique(X[:, i]) for i in range(self.n_atrib)]
        for clase, cantidad in zip(self.clases, cant_elem_clase):
            ej_pos = X[y == clase]
            self.prob_clase_atrib[clase] = {}
            for j in range(self.n_atrib):
                atributos_posibles = self.valores_atrib[j]
                # CON SUAVIZADO
                apariciones_condicionada = {element: (list(ej_pos[:, j]).count(element)+self.k)/(cantidad+self.k*self.cantidad_valores_atrib[j])
                                           for element in atributos_posibles}
                self.prob_clase_atrib[clase][f"Atrib_{j}"] = apariciones_condicionada

    def clasifica_prob(self, ejemplo):
        if self.clases is None:
            raise ClasificadorNoEntrenado("Clasificador aun no entrenado")
        else:
            probabilidades = []
            for clase in self.clases:
                diccionario = self.prob_clase_atrib[clase]
                acum = 1
                for i in range(self.n_atrib):
                    acum *= diccionario[f"Atrib_{i}"][ejemplo[i]]
                probabilidades.append(acum)
            probabilidades = self.prob_clases*probabilidades
            return {clase: prob for prob, clase in zip(probabilidades, self.clases)}

    def clasifica(self,ejemplo):
        predicciones = np.apply_along_axis(lambda fila: self.clasifica_prob(fila), axis=1, arr=ejemplo)
        # Al ser un diccionario no puedo usar apply otra vez
        predicciones_finales = np.array([max(pred, key=pred.get) for pred in predicciones])
        return predicciones_finales






  
# Ejemplo "jugar al tenis":
nb_tenis=NaiveBayes(k=0.5)
nb_tenis.entrena(X_tenis,y_tenis)

#print(nb_tenis.prob_clase_atrib)

ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
#(nb_tenis.clasifica_prob(ej_tenis))
# {'no': 0.7564841498559081, 'si': 0.24351585014409202}
# (nb_tenis.clasifica(ej_tenis))
# 'no'









# ----------------------------------------------
# 2.2) Implementación del cálculo de rendimiento
# ----------------------------------------------

# Definir una función "rendimiento(clasificador,X,y)" que devuelve la
# proporción de ejemplos bien clasificados (accuracy) que obtiene el
# clasificador sobre un conjunto de ejemplos X con clasificación esperada y.

def rendimiento(clasificador,X,y):
    predicciones = clasificador.clasifica(X)
    accuracy = np.mean(predicciones == y) # La media de True y Falses computa como 0 y 1
    return accuracy

# Ejemplo:

#(rendimiento(nb_tenis,X_tenis,y_tenis))
# 0.9285714285714286





# --------------------------
# 2.3) Aplicando Naive Bayes
# --------------------------

# Usando el clasificador Naive Bayes implementado, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Concesión de prestamos
# - Críticas de películas en IMDB 

# En todos los casos, será necesario separar un conjunto de test para dar la
# valoración final de los clasificadores obtenidos (ya realizado en el ejerciio 
# anterior). Ajustar también el valor del parámetro de suavizado k, usando un 
# conjunto de validación. 

# Describir (dejándolo comentado) el proceso realizado en cada caso, 
# y los rendimientos obtenidos. 

def estudio_naive_bayes(datos_X,datos_y):
    """
        Debido a la simpleza del clasificador Naive Bayes podemos automatizar este proceso. Los parámetros que
    deben buscarse son: El coeficiente de separación de los datos de entrenamiento y prueba, el cociente de separación
    de entrenamiento y validación y la constante de suavizado.
    Para los coeficientes de separación daremos los valores standard utilizados y vistos en clase. Para la constante de
    suavizado recorreremos una lista de valores. Además al ser un aprendizaje estocástico se podría entrenar al
    clasificador con el conjunto total, puesto que no hay riesgo de sobreajuste
    """
    porcentajes_separacion = [(0.30, 0.20), (0.40, 0.20)]
    clasificadores = []
    for i in range(2):
        porcentaje = porcentajes_separacion[i]
        X_ev,X_t,y_ev,y_t = particion_entr_prueba(datos_X,datos_y,porcentaje[0])
        X_e, X_v, y_e,y_v = particion_entr_prueba(X_ev,y_ev,porcentaje[1])
        for j in range(0,10):
            j = j/10
            clasificador = NaiveBayes(k=j)
            clasificador.entrena(X_e, y_e)
            accuracy = rendimiento(clasificador,X_v,y_v)
            print(f"Con una división de {porcentaje} y un suavizado de {j} se ha obtenido un rendimiento en el conjunto de validación de {accuracy}")
            clasificadores.append((clasificador,accuracy, porcentaje,j))
    for j in range(0, 10):
        j = j / 10
        clasificador = NaiveBayes(k=j)
        clasificador.entrena(datos_X, datos_y)
        accuracy = rendimiento(clasificador, datos_X, datos_y)
        print(f"El clasificador entrenado con todo el conjunto y un suavizado de {j} ha obtenido un rendimiento de {accuracy}")
        clasificadores.append((clasificador, accuracy,(0,0),j))
    ganador = max(clasificadores,key=(lambda clasif: clasif[1]))
    print(f"El mayor rendimiento en el conjunto de validación fue {ganador[1]}")
    print(f"Su rendimiento en el conjunto test es ´{rendimiento(ganador[0],X_t,y_t)}")
    print(f"Y sus hiperparámetros son k={ganador[3]}, y los porcentajes de division {ganador[2]}")
    return ganador[0]

def estudio_naive_bayes_imdb(min_suavizado, max_suavizado, salto):
    """
    Como en este caso ya tenemos hecha la division de entrenamiento y test obviamos ese paso en este estudio
    y automatizamos el proceso marcando los limites minimos y maximos para la constante de suavizado
    """
    clasificadores = []
    X_e, X_v, y_e, y_v = particion_entr_prueba(X_train_imdb, y_train_imdb,0.2)
    max_suavizado = int(max_suavizado*10)
    salto = int(salto*10)
    for k in range(min_suavizado,max_suavizado,salto):
        k = k/10
        clasificador = NaiveBayes(k=k)
        clasificador.entrena(X_train_imdb, y_train_imdb)
        accuracy = rendimiento(clasificador, X_e, y_e)
        print(f"Con un suavizado de {k} se ha obtenido un rendimiento en el conjunto de validación de {accuracy}")
        clasificadores.append((clasificador, accuracy,k))
    ganador = max(clasificadores, key=(lambda clasif: clasif[1]))
    print(f"El mayor rendimiento en el conjunto de validación fue {ganador[1]} con un suavizado de k={ganador[2]}")

#estudio_naive_bayes(X_votos,y_votos)
#estudio_naive_bayes(X_credito,y_credito)
#estudio_naive_bayes_imdb(0,1,0.1)
#-------------------------------------------------------------------------------------------------------
#                                        DATOS DE VOTOS
#       Despues de correr varias pruebas de la función previa se ha observado que la constante de
# suavizado suele ser numeros bastante bajos, k = 0 o k = 0.1. Por otro lado los porcentajes de división
# han sido 0.4 para el test y del 0.6 para entrenamiento 0.2 se ha reservado para validacion obteniendose
# asi un rendimiento entorno al 0.92.
#
#                                      DATOS DE PRESTAMOS
#       Despues de varias pruebas se ha comprobado que, para este conjunto de datos, el mejor clasificador
# entrenado con el total de los datos pero con una constante de suavizado k = 0.4, obteniendose un
# rendimiento entorno al 0.71.
#
#                                          DATOS IMDB
#       Como los datos de IMDB ya estaban segmentado en conjunto entrenamiento y test, no ha sido necesaria
# una division previa. Despues de multiples entrenamientos se ha observado que el rendimiento en el conjunto
# test ha sido de 0.84 independientemente del suavizado
#
#-----------------------------------------------------------------------------------------------------------

# ==================================
# EJERCICIO 3: NORMALIZADOR ESTÁNDAR
# ==================================


# Definir la siguiente clase que implemente la normalización "standard", es 
# decir aquella que traslada y escala cada característica para que tenga
# media 0 y desviación típica 1. 

# En particular, definir la clase: 


# class NormalizadorStandard():

#    def __init__(self):

#         .....
        
#     def ajusta(self,X):

#         .....        

#     def normaliza(self,X):

#         ......

# 

class NormalizadorStandard:
    def __init__(self):
        self.media = None
        self.deviacion_estandar = None


    def ajusta(self,X):
        self.medias = np.mean(X,axis=0)
        self.deviacion_estandar = np.std(X,axis=0)

    def normaliza(self, X):
        if self.medias is None:
            raise NormalizadorNoAjustado("Medias aun no calculadas")
        else:
            X_normalizado = (X - self.medias) / self.deviacion_estandar
            X[:] = X_normalizado  # [:] actualizamos el valor
            return X

# donde el método ajusta calcula las corresondientes medias y desviaciones típicas
# de las características de X necesarias para la normalización, y el método 
# normaliza devuelve el correspondiente conjunto de datos normalizados. 

# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

class NormalizadorNoAjustado(Exception): pass


# Por ejemplo:
    
    
normst_cancer=NormalizadorStandard()
normst_cancer.ajusta(Xe_cancer)

Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, la media y desviación típica de Xe_cancer_n deben ser 
# 0 y 1, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_n, 
# ni con Xp_cancer_n. 



# ------ 





























# ===========================================
# EJERCICIO 4: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# En este ejercicio se propone la implementación de un clasificador lineal 
# binario basado regresión logística (mini-batch), con algoritmo de entrenamiento 
# de descenso por el gradiente mini-batch (para minimizar la entropía cruzada).
# Diapositiva 50 del tema 3. 


# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
#                 batch_tam=64):

#         .....
        
#     def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
#                     early_stopping=False,paciencia=3):

#         .....        

#     def clasifica_prob(self,ejemplos):

#         ......
    
#     def clasifica(self,ejemplos):
                        
#          ......



# * El constructor tiene los siguientes argumentos de entrada:



#   + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#     durante todo el aprendizaje. Si rate_decay es True, rate es la
#     tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#   + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#     cada epoch. En concreto, si rate_decay es True, la tasa de
#     aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#     con la siguiente fórmula: 
#        rate_n= (rate_0)*(1/(1+n)) 
#     donde n es el número de epoch, y rate_0 es la cantidad introducida
#     en el parámetro rate anterior. Su valor por defecto es False. 
#  
#   + batch_tam: tamaño de minibatch


# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente. Las dos clases del problema 
#        son las que aparecen en el array y, y se deben almacenar en un atributo 
#        self.clases en una lista. La clase que se considera positiva es la que 
#        aparece en segundo lugar en esa lista.
#     
#     + Otros dos arrays Xv,yv, con los datos del conjunto de  validación, que se 
#       usarán en el caso de activar el parámetro early_stopping. Ambos con 
#       valor None por defecto. 

#     + n_epochs es el número máximo de epochs en el entrenamiento. 

#     + salida_epoch (False por defecto). Si es True, al inicio y durante el 
#       entrenamiento, cada epoch se imprime  el valor de la entropía cruzada 
#       del modelo respecto del conjunto de entrenamiento, y su rendimiento 
#       (proporción de aciertos). Igualmente para el conjunto de validación, si lo
#       hubiera. Esta opción puede ser útil para comprobar 
#       si el entrenamiento  efectivamente está haciendo descender la entropía
#       cruzada del modelo (recordemos que el objetivo del entrenamiento es 
#       encontrar los pesos que minimizan la entropía cruzada), y está haciendo 
#       subir el rendimiento.
# 
#     + early_stopping (booleano, False por defecto) y paciencia (entero, 3 por defecto).
#       Si early_stopping es True, dejará de entrenar cuando lleve un número de
#       epochs igual a paciencia sin disminuir la menor entropía conseguida hasta el momento
#       en el conjunto de validación 
#       NOTA: esto se suele hacer con un conjunto de validación, y mecanismo de 
#       "callback" para recuperar el mejor modelo, pero por simplificar implementaremos
#       esta versión más sencilla.  
#        



# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo 
#   asigna a cada ejemplo de pertenecer a la clase positiva.       
    


# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

class RegresionLogisticaMiniBatch():
    def __init__(self, rate=0.1, rate_decay=False, n_epochs=100, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam
        self.clases = None
        self.w = None

    def entrena(self, X, y, Xv = None, yv = None, n_epochs=100, salida_epochs=False, early_stopping=False, paciencia=3):
        self.clases = list(np.unique(y))
        self.w = np.random.rand(X.shape[1])*0.01 # Iniciamos los pesos con valores bajos para mayor estabilidad
        mu = self.rate
        if not (salida_epochs or early_stopping):
            for i in range(1, n_epochs+1):
                indices = np.random.randint(0,X.shape[0],size =self.batch_tam)
                X_b = X[indices]
                y_b = y[indices]
                predicciones = sigmoide(np.dot(X_b, self.w))
                error = y_b - predicciones
                gradiente = np.dot(X_b.T, error)
                self.w += mu * gradiente
                if self.rate_decay:
                    mu = self.rate / i
                if self.rate_decay:
                    mu = self.rate/i
        else:
            cont_parada = paciencia
            for i in range(1, n_epochs+1):
                indices = np.random.randint(0, X.shape[0], size =self.batch_tam)
                X_b = X[indices]
                y_b = y[indices]
                #Como es posible tener log(0) usamos un epsilon
                epsilon = 1e-15
                # EC en validacion
                predicciones_v = sigmoide(np.dot(Xv, self.w))
                predicciones_v = np.where(predicciones_v < epsilon, epsilon, predicciones_v)
                predicciones_v = np.where(predicciones_v > 1 - epsilon, 1 - epsilon, predicciones_v)
                log_pred_v = np.log(predicciones_v)
                compl_log_pred_v = np.log(1 - predicciones_v)
                n_entropia = (-np.sum(yv * log_pred_v + (1 - yv) * compl_log_pred_v))
                predicciones = sigmoide(np.dot(X_b, self.w))
                predicciones = np.where(predicciones < epsilon, epsilon, predicciones)
                predicciones = np.where(predicciones > 1 - epsilon, 1 - epsilon, predicciones)
                if early_stopping and i>=2:
                    if entropia <= int(n_entropia):
                        cont_parada -= 1
                        if not cont_parada:
                            print("Criterio de parada alcanzado: Early Stopping.")
                            break
                    else:
                        cont_parada = paciencia
                entropia = int(n_entropia)
                if salida_epochs:
                    # EC en entrenamiento
                    log_pred = np.log(predicciones)
                    compl_log_pred = np.log(1 - predicciones)
                    entropia_ent = (-np.sum(y_b * log_pred + (1 - y_b) * compl_log_pred))
                    print(f"La entropia en la iteración {i} es en Validación es: {n_entropia} ")
                    print(f"La entropia en la iteración {i} es en Entrenamiento es: {entropia_ent} ")
                error = y_b - predicciones
                gradiente = np.dot(X_b.T, error)
                self.w += mu * gradiente
                if self.rate_decay:
                    mu = self.rate / i


    def clasifica_prob(self,ejemplo):
        if self.clases is None:
            raise ClasificadorNoEntrenado("El clasificador aun no fue entrenado")
        return sigmoide(np.dot(ejemplo, self.w))

    def clasifica(self,ejemplo):
        arr = self.clasifica_prob(ejemplo)
        return (arr > 0.5)



  

# RECOMENDACIONES: 


# + IMPORTANTE: Siempre que se pueda, tratar de evitar bucles for para recorrer 
#   los datos, usando en su lugar funciones de numpy. La diferencia en eficiencia
#   es muy grande. 

# + Téngase en cuenta que el cálculo de la entropía cruzada no es necesario
#   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
#   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
#   cuando no sea necesario.     

# * Definir la función sigmoide usando la función expit de scipy.special, 
#   para evitar "warnings" por "overflow":

from scipy.special import expit

def sigmoide(x):
    return expit(x)

# * Usar np.where para definir la entropía cruzada. 

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


#lr_cancer=RegresionLogisticaMiniBatch(rate=0.1, rate_decay=True)

#lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer,salida_epochs=True)

#print(lr_cancer.clasifica(Xp_cancer_n[24:27]))
# array([0, 1, 0])   # Predicción para los ejemplos 24,25 y 26 

# >>> yp_cancer[24:27]
# array([0, 1, 0])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
# array([7.44297196e-17, 9.99999477e-01, 1.98547117e-18])


# Por ejemplo, los rendimientos sobre los datos (normalizados) del cáncer:

#(rendimiento(lr_cancer,Xe_cancer_n,ye_cancer))
# 0.9824561403508771

#(rendimiento(lr_cancer,Xp_cancer_n,yp_cancer))
# 0.9734513274336283




# Ejemplo con salida_epoch y early_stopping:

#lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

#lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# Inicialmente, en entrenamiento EC: 155.686323940485, rendimiento: 0.873972602739726.
# Inicialmente, en validación    EC: 43.38533009881579, rendimiento: 0.8461538461538461.
# Epoch 1, en entrenamiento EC: 32.7750241863029, rendimiento: 0.9753424657534246.
#          en validación    EC: 8.4952918658522,  rendimiento: 0.978021978021978.
# Epoch 2, en entrenamiento EC: 28.0583715052223, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.665719133490596, rendimiento: 0.967032967032967.
# Epoch 3, en entrenamiento EC: 26.857182744289368, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.09511082759361, rendimiento: 0.978021978021978.
# Epoch 4, en entrenamiento EC: 26.120803184993328, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.327991940213478, rendimiento: 0.967032967032967.
# Epoch 5, en entrenamiento EC: 25.66005010760342, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.376171724729662, rendimiento: 0.967032967032967.
# Epoch 6, en entrenamiento EC: 25.329200890122557, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.408704771704937, rendimiento: 0.967032967032967.
# PARADA TEMPRANA

# Nótese que para en el epoch 6 ya que desde la entropía cruzada obtenida en el epoch 3 
# sobre el conjunto de validación, ésta no se ha mejorado. 


# -----------------------------------------------------------------























# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando la regeresión logística implementada en el ejercicio 2, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros (tasa, rate_decay, batch_tam) para mejorar el rendimiento 
# (no es necesario ser muy exhaustivo, tan solo probar algunas combinaciones). 
# Usar para ello un conjunto de validación. 

# Dsctbir el proceso realizado en cada caso, y los rendimientos finales obtenidos
# sobre un conjunto de prueba (dejarlo todo como comentario)     


def estudio_logistica_mini_batch(datos_X,datos_y):
    """
        Para el estudio logistico se entrenaran varios clasificadores cada uno entrenado con parametros distintos
    para, despues de multiples pruebas, quedarnos con el que mejor resultado halla dado.
    Se volveran a utilizar los porcentajes de separacion vistos en clase. Para el rate de aprendizaje se utilizaran
    valores comprendidos en 0.1 y 1 y para el tamaño de batch se tomaran valores desde 10 a el tamaño total del
    conjunto de entrenamiento.
    """
    porcentajes_separacion = [(0.30, 0.20), (0.40, 0.20)]
    clasificadores = []
    for i in range(2):
        porcentaje = porcentajes_separacion[i]
        X_ev,X_t,y_ev,y_t = particion_entr_prueba(datos_X,datos_y,porcentaje[0])
        X_e,X_v,y_e,y_v = particion_entr_prueba(X_ev,y_ev,porcentaje[1])
        for j in range(1,11):
            j = j/10
            for rate_dec in [True,False]:
                for batch in range(10, X_e.shape[0],X_e.shape[0]//10):
                    clasificador = RegresionLogisticaMiniBatch(rate = j, rate_decay=rate_dec,batch_tam=batch)
                    clasificador.entrena(X_e, y_e)
                    accuracy = rendimiento(clasificador,X_v,y_v)
                    print(f"Con una división de {porcentaje} y un rate de {j} se ha obtenido un rendimiento en el conjunto de validación de {accuracy}")
                    clasificadores.append((clasificador,accuracy, porcentaje,j,rate_dec,batch))
    ganador = max(clasificadores,key=(lambda clasif: clasif[1]))
    print(f"El mayor rendimiento en el conjunto de validación fue {ganador[1]}")
    print(f"Su rendimiento en el conjunto test es ´{rendimiento(ganador[0],X_t,y_t)}")
    print(f"Y sus hiperparámetros son rate={ganador[3]}, porcentajes de division {ganador[2]}, rate_decay = {ganador[4]} y tamaño de batch = {ganador[5]}")
    return ganador[0]


def estudio_logistico_imdb():
    clasificadores = []
    X_e, X_v, y_e, y_v = particion_entr_prueba(X_train_imdb, y_train_imdb,0.2)

    for j in range(1, 11):
        j = j / 10
        for rate_dec in [True, False]:
            for batch in range(10, X_e.shape[0],X_e.shape[0]//3):
                clasificador = RegresionLogisticaMiniBatch(rate=j, rate_decay=rate_dec, batch_tam=batch)
                clasificador.entrena(X_e, y_e)
                accuracy = rendimiento(clasificador, X_v, y_v)
                print(
                    f"Con un rate de {j} se ha obtenido un rendimiento en el conjunto de validación de {accuracy}")
                clasificadores.append((clasificador, accuracy,  j, rate_dec, batch))
    ganador = max(clasificadores, key=(lambda clasif: clasif[1]))
    print(f"El mayor rendimiento en el conjunto de validación fue {ganador[1]}")
    print(f"Su rendimiento en el conjunto test es {rendimiento(ganador[0], X_test_imdb, y_test_imdb)}")
    print(f"Y sus hiperparámetros son rate={ganador[2]}, rate_decay = {ganador[3]} "
          f"y tamaño de batch = {ganador[4]}")
    return ganador[0]



#estudio_logistica_mini_batch(X_votos,y_votos)
#estudio_logistica_mini_batch(X_cancer,y_cancer)
#estudio_logistico_imdb()
#-----------------------------------------------------------------------------------------------
#
#                                       ESTUDIO VOTOS
#
#       En el estudio de los votos de los congresistas, despues de varias pruebas, se han
# obtenido los mejores resultados con una division del 40% para el conjunto test y un 20%
# para el de validacion. Con un rate bajo, alrededor de 0.2-0.4 y sin rate decay, con un
# batch_size bajo, sobre 10-30 ejemplos. Asi se ha obtenido un rendimiento de 0.98 - 1 en
# el conjunto de rendimiento y un 0.95-0.96 en el de test
#
#                                   ESTUDIO CANCER DE MAMA
#
#       Para el estudio del cancer de mama se han obtenido los mejores resultados con una
# division del 40% para el test y 20% para el conjunto de validacion, comenzando con un
# rate de 0.1, tamaño del batch de 10 y aplicando rate_decay. Asi se ha obtenido un
# rendimiento sobre el conjunto de validacion de 0.95 y resultando en un 0.91 en el test.
#
#                                       ESTUDIO IMDB
#
#       Como los datos de critica de cine en IMDB ya estaban separados por entrenamiento y
#  test solo lo hemos particionado para obtener un conjunto de validacion con un 20% del
#  conjunto test. Tras multiples pruebas se ha encontrado que los mejores resultado se han
#  obtenido con un rate alto, alrededor de 0.8-0.9, con rate_decay y un tamaño del batch
#  alrededor de 500. Asi se ha obtenido un rendimiento entorno al 0.8 tanto en el
#  conjunto de validacion como en el test.
#------------------------------------------------------------------------------------------------


























# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest. 


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que 
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica(self,ejemplos):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, aunque ahora referido a cada uno de los k entrenamientos a 
#  realizar (donde k es el número de clases) (
#  Por simplificar, supondremos que no hay conjunto de validación ni parada
#  temprana.  

class RL_OvR:
    def __init__(self,rate=0.1,rate_decay=False,batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.clasificadores = {}
        self.clases = None

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        self.clases = np.unique(y)
        for i in range(len(self.clases)):
            clasificador = RegresionLogisticaMiniBatch(self.rate, self.rate_decay, n_epochs, self.batch_tam)
            self.clasificadores[i] = clasificador
            clase = (y == self.clases[i])
            clasificador.entrena(X, clase, n_epochs, salida_epoch)

    def clasifica(self, X):
        if self.clases is None:
            raise ClasificadorNoEntrenado("El clasificador aun no fue entrenado")
        else:
            # Matriz para almacenar las predicciones de cada clasificador
            predicciones = np.zeros((X.shape[0], len(self.clases)))
            # Obtener las predicciones de cada clasificador
            for i, clasificador in self.clasificadores.items():
                predicciones[:, i] = clasificador.clasifica_prob(X)
            # Seleccionar la clase con la mayor probabilidad para cada ejemplo
            predicciones_finales = self.clases[np.argmax(predicciones, axis=1)]
            return predicciones_finales



#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)


rl_iris_ovr=RL_OvR(rate=0.001, batch_tam=8)

#rl_iris_ovr.entrena(Xe_iris, ye_iris)

#(rl_iris_ovr.clasifica(Xe_iris))

#(rendimiento(rl_iris_ovr,Xe_iris,ye_iris))
# 0.8333333333333334

#(rendimiento(rl_iris_ovr,Xp_iris,yp_iris))
# >>> 0.9
# --------------------------------------------------------------------




















# =====================================================
# EJERCICIO 7: APLICANDO LOS CLASIFICADORES MULTICLASE
# =====================================================


# -------------------------
# 8.1) Codificación one-hot
# -------------------------


# Los conjuntos de datos en los que algunos atributos son categóricos (es decir,
# sus posibles valores no son numéricos, o aunque sean numéricos no hay una 
# relación natural de orden entre los valores) no se pueden usar directamente
# con los modelos de regresión logística, o con redes neuronales, por ejemplo.

# En ese caso es usual transformar previamente los datos usando la llamada
# "codificación one-hot". Básicamente, cada columna se reemplaza por k columnas
# en los que los valores psoibles son 0 o 1, y donde k es el número de posibles 
# valores del atributo. El valor i-ésimo del atributo se convierte en k atributos
# (0 ...0 1 0 ...0 ) donde todas las posiciones son cero excepto la i-ésima.  

# Por ejemplo, sin un atributo tiene tres posibles valores "a", "b" y "c", ese atributo 
# se reemplazaría por tres atributos binarios, con la siguiente codificación:
# "a" --> (1 0 0)
# "b" --> (0 1 0)
# "c" --> (0 0 1)    

# Definir una función:    
    
#     codifica_one_hot(X) 

# que recibe un conjunto de datos X (array de numpy) y devuelve un array de numpy
# resultante de aplicar la codificación one-hot a X. Por simplificar supondremos
# que el array de entrada tiene todos sus atributos categóricos, y que por tanto 
# hay que codificarlos todos.

# NOTA: NO USAR PANDAS NI SKLEARN PARA ESTA FUNCIÓN
def codifica_one_hot(X):
    # Obtener el número de filas y columnas del array original
    n_clases, n_atrib = X.shape

    # Lista para almacenar el resultado final
    lista_one_hot_filas = []

    for feature_idx in range(n_atrib):
        # Obtener la columna actual
        column = X[:, feature_idx]
        # Encontrar los valores únicos y sus índices, unique los devuelve en orden inverso
        unique_values, inverse_indices = np.unique(column, return_inverse=True)
        # Crear una matriz de ceros con forma (n_samples, número de valores únicos)
        one_hot_col = np.zeros((n_clases, unique_values.size))
        # Colocar 1 en las posiciones correctas usando indices inversos
        # arange(n sample s) recorre las filas e inverse indices las columnas
        one_hot_col[np.arange(n_clases), inverse_indices] = 1
        # Añadir el array de codificación one-hot de la columna actual a la lista
        lista_one_hot_filas.append(one_hot_col)
    # Concatenar todos los arrays de codificación one-hot a lo largo de las columnas
    one_hot_encoded_array = np.hstack(lista_one_hot_filas)

    return one_hot_encoded_array
# Aplicar la función para obtener una codificación one-hot de los datos sobre
# concesión de prestamo bancario.     
#
# Xc=np.array([["a",1,"c","x"],
#                 ["b",2,"c","y"],
#                   ["c",1,"d","x"],
#                   ["a",2,"d","z"],
#                   ["c",1,"e","y"],
#                   ["c",2,"f","y"]])
   
# print(codifica_one_hot(Xc))
# 
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11     

    
  

























# ---------------------------------------------------------
# 8.2) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación OvR del ejercicio anterior y la de one-hot del
# apartado anterior, para obtener un clasificador que aconseje la concesión, 
# estudio o no concesión de un préstamo, basado en los datos X_credito, y_credito. 

# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado
# exhaustivo). Describirlo en los comentarios. 

X_credito_OH = codifica_one_hot(X_credito)

def estudio_credito(datos_X,datos_y):

    porcentajes_separacion = [(0.30, 0.20), (0.40, 0.20)]
    clasificadores = []
    for i in range(2):
        porcentaje = porcentajes_separacion[i]
        X_ev,X_t,y_ev,y_t = particion_entr_prueba(datos_X,datos_y,porcentaje[0])
        X_e,X_v,y_e,y_v = particion_entr_prueba(X_ev,y_ev,porcentaje[1])
        for j in range(1,11):
            j = j/10
            for rate_dec in [True,False]:
                for batch in range(10, X_e.shape[0],10):
                    clasificador = RL_OvR(rate = j, rate_decay=rate_dec, batch_tam=batch)
                    clasificador.entrena(X_e, y_e)
                    accuracy = rendimiento(clasificador,X_v,y_v)
                    print(f"Con una división de {porcentaje} y un rate de {j} se ha obtenido un rendimiento en el conjunto de validación de {accuracy}")
                    clasificadores.append((clasificador,accuracy, porcentaje,j,rate_dec,batch))
    ganador = max(clasificadores,key=(lambda clasif: clasif[1]))
    print(f"El mayor rendimiento en el conjunto de validación fue {ganador[1]}")
    print(f"Su rendimiento en el conjunto test es {rendimiento(ganador[0], X_t, y_t)}")
    print(f"Y sus hiperparámetros son rate={ganador[3]}, porcentajes de division {ganador[2]}, rate_decay = {ganador[4]} y tamaño de batch = {ganador[5]}")
    return ganador[0]


#estudio_credito(X_credito_OH,y_credito)

# ---------------------------------------------------------
#                   ESTUDIO DE CREDITO
#       Tras varias pruebas se ha obtenido que los mejores
# hiperparametros para el clasificador OneVSRest son:
# una división del 30% como test y del 70% restante
# dejar un 20% para validación. Un ratio de aprendizaje
# entorno al 0.8-0.9 y sin rate_decay. Para el tamaño del
# batch se han obtenido dos resultados distintos.Por un
# lado hemos obtenido batchs bastante bajos, con 50-60
# elementos, y por otro batch bastante grandes con unos
# 200-300 elementos. Asi, se consigue un rendimiento del
# 75% en el conjunto test y un 85 en el de validacion
# ---------------------------------------------------------
# 8.3) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación OvR del ejercicio anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en la 
#  carpeta datos/digitdata que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. 

# Se pide:
    
# * Definir las funciones auxiliares necesarias para cargar el dataset desde los 
#   archivos de texto, y crear variables:
#       X_entr_dg, y_entr_dg
#       X_val_dg, y_val_dg
#       X_test_dg, y_test_dg
#   que contengan arrays de numpy con el dataset proporcionado (USAR ESOS NOMBRES).  

def cargaImagenes(fichero, ancho, alto):
    def convierte_0_1(c):
        if c == " ":
            return 0
        else:
            return 1

    with open(fichero) as f:
        lista_imagenes = []
        ejemplo = []
        cont_lin = 0
        for lin in f:
            ejemplo.extend(list(map(convierte_0_1, lin[:ancho])))
            cont_lin += 1
            if cont_lin == alto:
                lista_imagenes.append(ejemplo)
                ejemplo = []
                cont_lin = 0
    return np.array(lista_imagenes)


def cargaClases(fichero):
    with open(fichero) as f:
        return np.array([int(c) for c in f])


trainingdigits = "datos/digitdata/trainingimages"

validationdigits = "datos/digitdata/validationimages"

testdigits = "datos/digitdata/testimages"

trainingdigitslabels = "datos/digitdata/traininglabels"

validationdigitslabels = "datos/digitdata/validationlabels"

testdigitslabels = "datos/digitdata/testlabels"

X_entr_dg = cargaImagenes(trainingdigits, 28, 28)

y_entr_dg = cargaClases(trainingdigitslabels)

X_valid_dg = cargaImagenes(validationdigits, 28, 28)

y_valid_dg = cargaClases(validationdigitslabels)

X_test_dg = cargaImagenes(testdigits, 28, 28)

y_test_dg = cargaClases(testdigitslabels)


# * Obtener un modelo de clasificación RL_OvR


# * Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
#   rate_decay para tratar de obtener un rendimiento aceptable (por encima del
#   75% de aciertos sobre test). 

# --------------------------------------------------------------------------
#
# clasificador = RL_OvR(rate=0.001,rate_decay=False,batch_tam=200)
# clasificador.entrena(X_entr_dg,y_entr_dg,n_epochs=1500)
# print(rendimiento(clasificador,X_test_dg,y_test_dg))
# Con estos hiperparámetros se consigue un rendimiento entorno al 0.83
# --------------------------------------------------------------------------





















# ********************************************************************************
# ********************************************************************************
# ********************************************************************************
# ********************************************************************************

# EJEMPLOS DE PRUEBA

# LAS SIGUIENTES LLAMADAS SERÁN EJECUTADAS POR EL PROFESOR EL DÍA DE LA PRESENTACIÓN.
# UNA VEZ IMPLEMENTADAS LAS DEFINICIONES Y FUNCIONES (INCLUIDAS LAS AUXILIARES QUE SE
# HUBIERAN NECESITADO) Y REALIZADOS LOS AJUSTES DE HIPERPARÁMETROS, 
# DEJAR COMENTADA CUALQUIER LLAMADA A LAS FUNCIONES QUE SE TENGA EN ESTE ARCHIVO 
# Y DESCOMENTAR LAS QUE VIENE A CONTINUACIÓN.

# EN EL APARTADO FINAL DE RENDINIENTOS FINALES, USAR LA MEJOR COMBINACIÓN DE 
# HIPERPARÁMETROS QUE SE HAYA OBTENIDO EN CADA CASO, EN LA FASE DE AJUSTE. 

# ESTE ARCHIVO trabajo-1-iacd-23-24.py SERA CARGADO POR EL PROFESOR, 
# TENIENDO EN LA MISMA CARPETA LOS ARCHIVOS OBTENIDOS
# DESCOMPRIMIENDO datos-trabajo-1-iacd.zip.
# ES IMPORTANTE QUE LO QUE SE ENTREGA SE PUEDA CARGAR SIN ERRORES Y QUE SE EJECUTEN LOS 
# EJEMPLOS QUE VIENEN A CONTINUACIÓN. SI ALGUNO DE LOS EJERCICIOS NO SE HA REALIZADO 
# O DEVUELVE ALGÚN ERROR, DEJAR COMENTADOS LOS CORRESPONDIENTES EJEMPLOS. 



# *********** DESCOMENTAR A PARTIR DE AQUÍ

# print("************ PRUEBAS EJERCICIO 1:")
# print("**********************************\n")
# Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)
# print("Partición votos: ",y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0])
# print("Proporción original en votos: ",np.unique(y_votos,return_counts=True))
# print("Estratificación entrenamiento en votos: ",np.unique(ye_votos,return_counts=True))
# print("Estratificación prueba en votos: ",np.unique(yp_votos,return_counts=True))
# print("\n")

# Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
# print("Proporción original en cáncer: ", np.unique(y_cancer,return_counts=True))
# print("Estratificación entr-val en cáncer: ",np.unique(yev_cancer,return_counts=True))
# print("Estratificación prueba en cáncer: ",np.unique(yp_cancer,return_counts=True))
# Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
# print("Estratificación entrenamiento cáncer: ", np.unique(ye_cancer,return_counts=True))
# print("Estratificación validación cáncer: ",np.unique(yv_cancer,return_counts=True))
# print("\n")

# Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)
# print("Estratificación entrenamiento crédito: ",np.unique(ye_credito,return_counts=True))
# print("Estratificación prueba crédito: ",np.unique(yp_credito,return_counts=True))
# print("\n\n\n")





# print("************ PRUEBAS EJERCICIO 2:")
# print("**********************************\n")

# nb_tenis=NaiveBayes(k=0.5)
# nb_tenis.entrena(X_tenis,y_tenis)
# ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
# print("NB Clasifica_prob un ejemplo tenis: ",nb_tenis.clasifica_prob(ej_tenis))
# print("NB Clasifica un ejemplo tenis: ",nb_tenis.clasifica([ej_tenis]))
# print("\n")

# nb_votos=NaiveBayes(k=1)
# nb_votos.entrena(Xe_votos,ye_votos)
# print("NB Rendimiento votos sobre entrenamiento: ", rendimiento(nb_votos,Xe_votos,ye_votos))
# print("NB Rendimiento votos sobre test: ", rendimiento(nb_votos,Xp_votos,yp_votos))
# print("\n")


# nb_credito=NaiveBayes(k=1)
# nb_credito.entrena(Xe_credito,ye_credito)
# print("NB Rendimiento crédito sobre entrenamiento: ", rendimiento(nb_credito,Xe_credito,ye_credito))
# print("NB Rendimiento crédito sobre test: ", rendimiento(nb_credito,Xp_credito,yp_credito))
# print("\n")


# nb_imdb=NaiveBayes(k=1)
# nb_imdb.entrena(X_train_imdb,y_train_imdb)
# print("NB Rendimiento imdb sobre entrenamiento: ", rendimiento(nb_imdb,X_train_imdb,y_train_imdb))
# print("NB Rendimiento imdb sobre test: ", rendimiento(nb_imdb,X_test_imdb,y_test_imdb))
# print("\n")


# print("************ PRUEBAS EJERCICIO 3:")
# print("**********************************\n")



# normst_cancer=NormalizadorStandard()
# normst_cancer.ajusta(Xe_cancer)
# Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
# Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
# Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

# print("Normalización cancer entrenamiento: ",np.mean(Xe_cancer,axis=0))
# print("Normalización cancer validación: ",np.mean(Xv_cancer,axis=0))
# print("Normalización cancer test: ",np.mean(Xp_cancer,axis=0))

# print("\n\n\n")



# print("************ PRUEBAS EJERCICIO 4:")
# print("**********************************\n")


# lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
# lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)
# print("LR clasifica cuatro ejemplos cáncer (y valor esperado): ",lr_cancer.clasifica(Xp_cancer_n[17:21]),yp_cancer[17:21])
# print("LR clasifica_prob cuatro ejemplos cáncer: ", lr_cancer.clasifica_prob(Xp_cancer_n[17:21]))
# print("LR rendimiento cáncer entrenamiento: ", rendimiento(lr_cancer,Xe_cancer_n,ye_cancer))
# print("LR rendimiento cáncer prueba: ", rendimiento(lr_cancer,Xp_cancer_n,yp_cancer))

# print("\n\n CON SALIDA Y EARLY STOPPING**********************************\n")

# lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
# lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# print("\n\n\n")

# print("************ PRUEBAS EJERCICIO 6:")
# print("**********************************\n")

# Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=16)

# rl_iris_ovr.entrena(Xe_iris,ye_iris)

# print("OvR Rendimiento entrenamiento iris: ",rendimiento(rl_iris_ovr,Xe_iris,ye_iris))
# print("OvR Rendimiento prueba iris: ",rendimiento(rl_iris_ovr,Xp_iris,yp_iris))
# print("\n\n\n")



# print("************ RENDIMIENTOS FINALES REGRESIÓN LOGÍSTICA EN CRÉDITO, IMDB y DÍGITOS")
# print("*******************************************************************************\n")


# # ATENCIÓN: EN CADA CASO, USAR LA MEJOR COMBINACIÓN DE HIPERPARÁMETROS QUE SE HA 
# # DEBIDO OBTENER EN EL PROCESO DE AJUSTE

# print("==== MEJOR RENDIMIENTO RL SOBRE VOTOS:")
# RL_VOTOS=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,batch_tam=64) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
# RL_VOTOS.entrena(Xe_votos,ye_votos) # Aumentar o disminuir los epochs si fuera necesario
# print("Rendimiento RL entrenamiento sobre votos: ",rendimiento(RL_VOTOS,Xe_votos,ye_votos))
# print("Rendimiento RL test sobre votos: ",rendimiento(RL_VOTOS,Xp_votos,yp_votos))
# print("\n")


# print("==== MEJOR RENDIMIENTO RL SOBRE CÁNCER:")
# RL_CANCER=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,batch_tam=64) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
# RL_CANCER.entrena(Xe_cancer,ye_cancer) # Aumentar o disminuir los epochs si fuera necesario
# print("Rendimiento RL entrenamiento sobre cáncer: ",rendimiento(RL_CANCER,Xe_cancer,ye_cancer))
# print("Rendimiento RL test sobre cancer: ",rendimiento(RL_CANCER,Xp_cancer,yp_cancer))
# print("\n")


# print("==== MEJOR RENDIMIENTO RL_OvR SOBRE CREDITO:")
# X_credito_oh=codifica_one_hot(X_credito)
# Xe_credito_oh,Xp_credito_oh,ye_credito,yp_credito=particion_entr_prueba(X_credito_oh,y_credito,test=0.3)

# RL_CLASIF_CREDITO=RL_OvR(rate=0.1,rate_decay=True,batch_tam=64) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
# RL_CLASIF_CREDITO.entrena(Xe_credito_oh,ye_credito) # Aumentar o disminuir los epochs si fuera necesario
# print("Rendimiento RLOVR  entrenamiento sobre crédito: ",rendimiento(RL_CLASIF_CREDITO,Xe_credito_oh,ye_credito))
# print("Rendimiento RLOVR  test sobre crédito: ",rendimiento(RL_CLASIF_CREDITO,Xp_credito_oh,yp_credito))
# print("\n")


# print("==== MEJOR RENDIMIENTO RL SOBRE IMDB:")
# RL_IMDB=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,batch_tam=64) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
# RL_IMDB.entrena(X_train_imdb,y_train_imdb) # Aumentar o disminuir los epochs si fuera necesario
# print("Rendimiento RL entrenamiento sobre imdb: ",rendimiento(RL_IMDB,X_train_imdb,y_train_imdb))
# print("Rendimiento RL test sobre imdb: ",rendimiento(RL_IMDB,X_test_imdb,y_test_imdb))
# print("\n")


# print("==== MEJOR RENDIMIENTO RL SOBRE DIGITOS:")
# RL_DG=RL_OvR(rate=0.1,rate_decay=True,batch_tam=64) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
# RL_DG.entrena(X_entr_dg,y_entr_dg) # Aumentar o disminuir los epochs si fuera necesario
# print("Rendimiento RL entrenamiento sobre dígitos: ",rendimiento(RL_DG,X_entr_dg,y_entr_dg))
# print("Rendimiento RL validación sobre dígitos: ",rendimiento(RL_DG,X_val_dg,y_val_dg))
# print("Rendimiento RL test sobre dígitos: ",rendimiento(RL_DG,X_test_dg,y_test_dg))








