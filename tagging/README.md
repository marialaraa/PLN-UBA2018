#Práctico 2


## Ejercicio 1
Se observa a continuación una tabla con las diez etiquetas más frecuentes
e información sobre cada una.

| Etiqueta | Frecuencia | Porcentaje del total | Palabras más frecuentes |Significado|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|sp000	| 79884 | 15.45	| de, en, a, del, con | Preposiciones. |
|nc0s000| 63452 | 12.27	| presidente, equipo, partido, país, año | Sustantivos comunes que estén en singular. |
|da0000	| 54549 | 10.55	| la, el, los, las, El | Artículos. |
|aq0000	| 33906 | 6.56	| pasado, gran, mayor, nuevo, próximo | Adjetivos descriptivos. |
|fc     | 30147 | 5.83	| , | La coma. |
|np00000| 29111 | 5.63	| Gobierno, España, PP, Barcelona, Madrid | Sustantivos propio. |
|nc0p000| 27736 | 5.36	| años, millones, personas, países, días | Sustantivos comunes que estén en plural. |
|fp	    | 17512 | 3.39	| . | El punto. |
|rg	    | 15336	| 2.97	| más, hoy, también, ayer, ya | Adverbios. |
|cc	    | 15023	| 2.90	| y, pero, o, Pero, e | Conjunciones. |

| Nivel de ambigüedad | Cantidad de palabras | Porcentaje del total | Palabras más frecuentes |
|:-----:|:-----:|:-----:|:-----:|
|1|	43972|	94.56	|,, con, por, su, El|
|2|	2318|	4.98	|el, en, y, ", los|
|3|	180|	0.39	|de, la, ., un, no|
|4|	23|	0.05	|que, a, dos, este, fue|
|5|	5|	0.01	|mismo, cinco, medio, ocho, vista|
|6|	3|	0.01	|una, como, uno|

Se muestra hasta el nivel 6 de ambigüedad, ya que no hay niveles más
altos en el corpus presentado.

## Ejercicio 2
Se define una clase `BaselineTagger`, que definirá etiquetas de acuerdo
a los datos de entrenamiento. Se entrena y evalúa el modelo utilizando
el corpus de Ancora en español (http://clic.ub.edu/corpus/es/ancora).
Luego de entrenar y evaluar el corpus, podemos observa una precisión
de 87.58% en total. Con respecto a las palabras conocidas (es decir,
se encontraban en el corpus de entrenamiento y evaluación) el
porcentaje de precisión es de 95.27%. Para las palabras desconocidas
la exactitud en el etiquetado es muy bajo (18.01%).

## Ejercicio 3
Se implementan distintos features para luego ser utilizados en la
vectorización. Se definen los siguientes métodos:

    * word_lower: indica si la palabra está en minúsculas en su totalidad.
    * word_istitle: indica si la palabra empieza en mayúscula.
    * word_isupper: indica si la palabra actual está en mayúsculas en su totalindad.
    * word_isdigit: la palabra actual es un número o no.
    * NPrevTags: para un `n` dado indica la tupla de los últimos `n` tags.
    * PrevWord: para un feature `f` dado, aplicalo sobre la palabra anterior a la actual.

## Ejercicio 4
En este ejercicio se implementa un Maximum Entropy Markov Model. Se utilizará
`Vectorizer` con los features definidos en el ejercicio anterior y como clasificador
de máxima entropia a `LogisticRegression`.

