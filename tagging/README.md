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

| n | Precisión total | Precisión para palabras conocidas | Precisión para palabras desconocidas | Clasificador utilizado | Tiempo de entrenamiento (min:seg) | Tiempo de evaluación (min:seg) | Features utilizados |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 1 | 91.69% | 95.00% | 61.69% | LogisticRegression | 16:00 | 00:55 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit, PrevWord(word_lower), PrevWord(word_istitle), PrevWord(word_isdigit), NextWord(word_lower) |
| 2 | 91.24% | 94.57% | 61.05% | LogisticRegression | 14:23 | 00:41 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit, PrevWord(word_lower), PrevWord(word_istitle), PrevWord(word_isdigit), NextWord(word_lower), NPrevTags(1) |
| 3 | 91.42% | 94.67% | 61.98% | LogisticRegression | 19:26 | 00:45 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit, PrevWord(word_lower), PrevWord(word_istitle), PrevWord(word_isdigit), NextWord(word_lower), NPrevTags(1), NPrevTags(2) |
| 4 | 91.45% | 94.67% | 62.27% | LogisticRegression | 20:00 | 01:05 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit, PrevWord(word_lower), PrevWord(word_istitle), PrevWord(word_isdigit), NextWord(word_lower), NPrevTags(1), NPrevTags(2), NPrevTags(3) |
| 1 | 87.49% | 93.30% | 34.93% | LogisticRegression | 05:03 | - | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |
| 2 | 89.40% | 93.69% | 50.61% | LogisticRegression | 07:32 | - | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |
| 3 | 89.27% | 94.05% | 45.98% | LogisticRegression | 09:11 | - | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |
| 4 | 89.15% | 94.08% | 44.57% | LogisticRegression | 10:18 | - | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |
| 1 | 84.24% | 91.53% | 18.23% | MultinomialNB | 00:34 | 30:40 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |
| 2 | 79.86% | 85.66% | 27.30% | MultinomialNB | 00:38 | 40:00 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |
| 3 | 79.51% | 85.52% | 25.06% | MultinomialNB | 00:50 | 42:25 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |
| 4 | 77.95% | 83.78% | 25.15% | MultinomialNB | 00:50 | 44:10 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |
| 1 | 89.40% | 95.24% | 36.53% | LinearSVC | 06:43 | 00:38 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |
| 2 | 92.48% | 96.91% | 52.39% | LinearSVC | 04:44 | 00:37 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |
| 3 | 91.89% | 96.81% | 47.41% | LinearSVC | 05:06 | 00:50 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |
| 4 | 91.49% | 96.61% | 45.22% | LinearSVC | 08:09 | 00:57 | word_lower, prev_tags, word_istitle, word_isupper, word_isdigit |

Quiero destacar que muchos de los entrenamientos y evaluaciones se ejecutaron
en paralelo, por lo que temporalmente no se podrían comparar. Hice esto
ya que para algunos clasificadores el tiempo de entrenamiento o evaluación
es muy alto. En el caso del clasificador de regresión logística la etapa
de entrenamiento requiere mucho tiempo, sobre todo para corpus grandes.
Por otro lado, en el caso del clasificador multinomial Naive Bayes requiere
de mucho tiempo en la etapa de evaluación.