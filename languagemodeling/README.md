# Práctico 1

## Ejercicio 1: Corpus
Se elige cargar el corpus de una saga de novelas llamado Outlander.
Este libro se caracteriza por tener personajes de distintas épocas por
lo que los personajes deberían usar palabras distintas según el tiempo
en el que se encuentren. Esto amplia el vocabulario que se presenta
en el corpus de datos. El corpus se encuentra en inglés. Se presentan
los 7 libros de la saga, que en total suman 16,6 GB de texto.

Se editó el script `train.py` para que utilice el corpus definido.
No se necesitó modificar la expresión regular que parsea los tokens
ya que el texto es simple sin ningún símbolo o condición especial.


## Ejercicio 2: Modelo de n-gramas
Se implementa una clase N-Gramas con indicadores de comienzo y fin de
oración (`<s>` y `</s>` respectivamente) y permite indicar el `n` para
el tipo de modelo que se quiere utilizar. En la creación de la clase
se generan diccionaros que contienen la cantidad de apariciones de
combinaciones de tokens para aquellos de tamaño `n` y `n-1`. Para ello,
se recorre la oración con una ventana de tamaño `n` y `n-1` y se
contabiliza cada una.

La clase NGram tiene funciones que devuelven la probabilidad de aparición
de un token para el corpus indicado. Por posibles errores de overflow
se presenta también el método probabilistico utilizando logaritmo.

## Ejercicio 3: Generación de Texto
La clase `NGramGenerator` permite generar oraciones a partir de un
modelo de `NGram` para un corpus determinado. Se presentan a continuación
ejemplos de oraciones generadas para distintos tamaños de n-gramas.

### N = 1 / Unigramas
* to ?” is be wiped ; heading , he men m across . looked was
* head take had grubby was said eyes these I reflex for ’
* quiet but bunched to the sometimes here It , . grandeur
* the stood French the rimmed more her not ’
### N = 2 / Bigramas
* I think it .
* “ Do nay affection .
* “ Broom - splitting logs and cattail reeds , but uplifting .
* A moment , and he said , looking the trapped hummingbird .
### N = 3 / Trigramas
* Even Duncan , who appeared to have medical care , but had some .
* Clean , though ?”
* And — it would be leaving in their sockets , stinging and blurring in my time was ripe to have seen those hands cracked against his chest , struggling to get any in Scotland .
* A soft thump of small shorebirds were running down the hill .
### N = 4 / Cuatrigramas
* Where the next fellow she meets is you ,” he said , and looked at Lord Lovat , had not referred to his daughter ’ s happiness somehow through my eyes .
* Tuscarora ’ s more picturesque Gaelic phrases , which she waved under his nose , and shrugged .
* Only small , cool observations that poked their heads above the surface .
* “ One you know is yours ?” I asked , peering over my shoulder as we entered the hallway , also looking worried .

## Ejercicio 4: Suavizado "add-one"
Se agrega una nueva clase denominada `AddOneNGram`. Se requiere
de un diccionario con la cantidad de apariciones para los tokens que
correspondan según el `n` indicado en el modelo. Además, se calcula
el tamaño del conjunto de palabras que aparecen en el corpus de datos.
Este último valor se utilizará en el cálculo de la probabilidad
para un token dado y su tokens previos.

## Ejercicio 5: Evaluación de Modelos de Lenguaje
Antes de evaluar los modelos de lenguaje, se implementó una modificación
en el script `train.py` y `eval.py` para dividir el corpus en datos
de training y de test. En este caso particular, para el corpus planteado,
se decidió usar el primer libro de la saga como test y los restantes
6 libros como datos de training.

Se observan los siguientes resultados para el modelo de AddOneNGram
para n={1,2,3,4}.

| N | Logarithmic probability | Cross entropy  | Perplexity |
| :-------------: | :-------------: | :-------------: | :-------------: |
| 1 | -3853485.08 | 9.36 | 658.20  |
| 2 | -4028672.09 | 9.78 | 884.07 |
| 3 | -5234911.68 | 12.71 | 6740.74 |
| 4 | -5728369.84 | 13.91 | 15474.36 |

Cuanto más bajo es el valor de `perplexity`, entonces mejor será el modelo
del lenguaje en el modelado de oraciones no vistas previamente. Se observa
una mejora importante en el modelo de add-one unigrama contra
el modelo de add-one cuatrigramas.

<!--1gram-->
<!--sobre el mismo corpus de entrenamiento-->
<!--Log probability: -31703639.881051958-->
<!--Cross entropy: 9.271691331130986-->
<!--Perplexity: 618.0978048404776-->

<!--4gram-->
<!--sobre el mismo corpus de entrenamiento-->
<!--Log probability: -44854392.5005444-->
<!--Cross entropy: 13.117613109118027-->
<!--Perplexity: 8887.815228771982-->

## Ejercicio 6: Evaluación de Modelos de Lenguaje
Se implementa el suavizado por interpolación como un nuevo modelo del
lenguaje. La clase `InterpolatedNGram` presenta los métodos necesarios.
Al inicializar la clase se separa el corpus en training data y test data
en caso de que se deba seleccionar un gamma. Luego, se generan los modelos
de n-gramas para todo `n` menor al valor pasado al modelo. En caso de
que se utilice AddOne para los unigramas, entonces se genera el modelo
necesario.

Si no se indica el valor de `gamma` que se deba utilizar, entonces
se realiza un barrido para seleccionar el mejor para el corpus utilizado.

