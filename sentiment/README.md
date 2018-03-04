# Práctico 3

## Ejercicio 1: Corpus de Tweets: Estadísticas Básicas
Se genera el archivo `stats.py` que mostrará estadísticas sobre el corpus
de tweets de `InterTass` y `GeneralTASS`. Se muestra a continuación el
resultado que se obtiene de ejecutar el script.

```
Estadísticas de InterTass
================
Cantidad total de tweets: 1008
Cantidad de tweets con polaridad P: 318
Cantidad de tweets con polaridad N: 418
Cantidad de tweets con polaridad NEG: 133
Cantidad de tweets con polaridad NONE: 139
================


Estadísticas de GeneralTass
================
Cantidad total de tweets: 7219
Cantidad de tweets con polaridad P: 2884
Cantidad de tweets con polaridad N: 2182
Cantidad de tweets con polaridad NEG: 670
Cantidad de tweets con polaridad NONE: 1483
================
```

## Ejercicio 2: Mejoras al Clasificador Básico de Polaridad
En este ejercicio se implementan cuatro mejoras en el clasificador de
sentimientos ya implementado. Se decidió implementar las siguientes:

* **Mejor Tokenizer**: Se resuelve utilizando el tokenizer para tweets
del NLTK.
* **Normalización Básica de Tweets**: Antes de entrenar el corpus, se
_limpian_ las oraciones eliminando las referencias a otros usuarios,
las urls y las repeticiones de vocales.
* **Filtrado de stopwords**: Se setea el `CountVectorizer` para que ignore
stopwords del castellano. Se utiliza el que provee NLTK.
* **Stemming**: Se modifica el tokenizador del `CountVectorizer` para que
haga stemming sobre las palabras. Se utiliza el Snowball stemmer que provee
NLTK. 

Veamos que ocurra al implementar cada una de las mejoras por separado
y al implementar todas en conjunto.

Resultados de la curva de aprendizaje para los tres clasificadores.
### Mejor Tokenize
#### mnb
n=64, acc=47.63, f1=28.65
n=128, acc=48.22, f1=37.63
n=257, acc=51.19, f1=39.84
n=514, acc=54.94, f1=42.79
n=1028, acc=54.94, f1=42.66
n=2056, acc=53.36, f1=43.85
n=4113, acc=51.98, f1=45.76
n=8227, acc=52.77, f1=39.88

#### maxent
n=64, acc=43.68, f1=29.85
n=128, acc=46.25, f1=30.47
n=257, acc=49.01, f1=34.33
n=514, acc=48.81, f1=34.62
n=1028, acc=54.15, f1=40.60
n=2056, acc=54.15, f1=43.39
n=4113, acc=52.37, f1=41.84
n=8227, acc=51.98, f1=41.03

#### svm
n=64, acc=42.29, f1=30.32
n=128, acc=43.87, f1=30.60
n=257, acc=44.07, f1=30.94
n=514, acc=46.64, f1=34.06
n=1028, acc=50.00, f1=38.81
n=2056, acc=50.79, f1=41.08
n=4113, acc=49.21, f1=39.18
n=8227, acc=48.42, f1=38.20

### Normalización Básica de Tweets
#### mnb
n=64, acc=46.64, f1=26.15
n=128, acc=49.60, f1=44.32
n=257, acc=51.58, f1=40.05
n=514, acc=54.94, f1=42.74
n=1028, acc=55.34, f1=43.04
n=2056, acc=55.14, f1=49.19
n=4113, acc=55.93, f1=48.20
n=8227, acc=55.14, f1=51.28

#### maxent
n=64, acc=48.22, f1=30.86
n=128, acc=49.01, f1=34.52
n=257, acc=50.40, f1=35.32
n=514, acc=51.19, f1=38.71
n=1028, acc=54.94, f1=39.45
n=2056, acc=54.35, f1=41.65
n=4113, acc=53.95, f1=43.37
n=8227, acc=51.38, f1=40.91

#### svm
n=64, acc=49.41, f1=34.05
n=128, acc=47.04, f1=35.10
n=257, acc=47.23, f1=34.16
n=514, acc=49.41, f1=39.31
n=1028, acc=52.96, f1=41.61
n=2056, acc=51.19, f1=41.10
n=4113, acc=51.38, f1=39.65
n=8227, acc=49.41, f1=39.45

### Filtrado de stopwords
#### maxent
n=64, acc=44.47, f1=25.23
n=128, acc=44.86, f1=25.37
n=257, acc=48.62, f1=32.28
n=514, acc=51.98, f1=36.85
n=1028, acc=52.17, f1=36.24
n=2056, acc=49.80, f1=38.22
n=4113, acc=48.42, f1=41.40
n=8227, acc=50.40, f1=40.55

#### maxent
n=64, acc=47.63, f1=42.31
n=128, acc=48.22, f1=43.16
n=257, acc=49.01, f1=40.59
n=514, acc=51.19, f1=28.87
n=1028, acc=51.98, f1=34.46
n=2056, acc=53.36, f1=40.64
n=4113, acc=50.79, f1=38.81
n=8227, acc=51.38, f1=40.40

#### svm
n=64, acc=47.43, f1=33.02
n=128, acc=46.64, f1=28.83
n=257, acc=47.43, f1=32.31
n=514, acc=49.80, f1=39.74
n=1028, acc=49.60, f1=39.35
n=2056, acc=48.81, f1=37.93
n=4113, acc=47.04, f1=36.96
n=8227, acc=47.63, f1=38.57

### Stemming
#### mnb
n=64, acc=44.86, f1=36.91
n=128, acc=47.83, f1=37.40
n=257, acc=50.40, f1=39.07
n=514, acc=54.15, f1=41.92
n=1028, acc=53.95, f1=31.62
n=2056, acc=55.73, f1=36.59
n=4113, acc=55.53, f1=40.79
n=8227, acc=54.15, f1=43.55

#### maxent
n=64, acc=46.44, f1=33.08
n=128, acc=43.08, f1=26.92
n=257, acc=48.42, f1=28.79
n=514, acc=50.99, f1=36.07
n=1028, acc=51.38, f1=37.69
n=2056, acc=54.35, f1=41.58
n=4113, acc=55.93, f1=46.16
n=8227, acc=51.98, f1=41.21

#### svm
n=64, acc=45.26, f1=33.56
n=128, acc=42.29, f1=29.20
n=257, acc=45.06, f1=30.82
n=514, acc=48.02, f1=35.67
n=1028, acc=48.02, f1=35.85
n=2056, acc=51.19, f1=38.82
n=4113, acc=53.75, f1=42.47
n=8227, acc=52.77, f1=42.07

### Todos las mejoras en conjunto
#### mnb
n=64, acc=46.25, f1=27.84
n=128, acc=45.45, f1=29.64
n=257, acc=49.60, f1=28.47
n=514, acc=52.17, f1=35.15
n=1028, acc=52.77, f1=35.41
n=2056, acc=53.16, f1=41.23
n=4113, acc=51.78, f1=42.69
n=8227, acc=52.57, f1=37.48

#### maxent
n=64, acc=49.60, f1=40.97
n=128, acc=47.63, f1=33.37
n=257, acc=50.00, f1=31.64
n=514, acc=50.99, f1=35.08
n=1028, acc=52.37, f1=37.68
n=2056, acc=53.56, f1=40.40
n=4113, acc=53.56, f1=43.73
n=8227, acc=52.57, f1=41.90

#### svm
n=64, acc=46.44, f1=28.21
n=128, acc=45.26, f1=34.04
n=257, acc=45.85, f1=33.58
n=514, acc=47.63, f1=37.53
n=1028, acc=45.85, f1=35.60
n=2056, acc=48.42, f1=39.60
n=4113, acc=50.20, f1=42.25
n=8227, acc=48.81, f1=41.19

<!--Resultado de la evaluación sobre el corpus de development de InterTASS. Usar el script eval.py.-->
