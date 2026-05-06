# Multilayer Perceptron (MLP) - Implementaciones desde Cero (From Scratch)

Este repositorio contiene un proyecto de aprendizaje personal sobre los fundamentos matemáticos y de programación detrás de las Redes Neuronales Artificiales, específicamente el **Multilayer Perceptron (MLP)**. Se han codificado desde cero usando únicamente **Python** y **NumPy**.

A lo largo del proyecto, se presentan dos enfoques distintos para resolver el mismo problema (como el problema lógico del XOR), también se incluyeron problemas clásicos de la literatura como la clasificacion multietiqueta de Iris y la clasificación binaria de la nube de puntos centrada, documentando la evolución desde una arquitectura conceptual hasta una aproximación algebraica más profesional.

---

[McCulloch y Pitts en 1943](https://es.wikipedia.org/wiki/Neurona_de_McCulloch-Pitts) proponen un simple modelo de neurona biológica, cual mas tarde sería conocido como Neurona Artificial, este se creo para tareas binarias (on/off) de entrada y salida. La  
propuesta de la neurona era el cálculo de compuertas lógicas

![](./assets/image%208.png)

# Perceptrón:

El Perceptron se puede resumir como la estructura mas simple de una RNA. Inventado en 1957 por Frank Rosenblatt. Está basado en un modelo de Neurona Artificial, llamada unidad del umbral lógico (TLU), o a veces Unidad Lineal del Umbral (LTU). Las entradas y salidas son numericos o valores binarios, y cada entrada esta conectada con un peso ($w$).

El TLU primero calcula un función lineal como una suma ponderada de los pesos por las variables de entrada mas un término de sesgo. Entonces se aplica una función de activación, que es quien estructura de forma no lineal la salida. Así como la regresión logística los parámetros del modelo son la entrada del peso y el término bias o sesgo $b$.

## ![](./assets/image%201%205.png)

### Modelación:

Partimos de:

- pesos (aleatorios) -> Fórmula de Xavier $w∼N(0,\frac{1}​{\sqrt{n}}​)$
- learning_rate -> pasos de aprendizaje
- bias -> Siempre se inicializa en cero

Ajuste de pesos (Perceptron Learning Rules):

- pesos -> peso + delta_peso
- bias -> bias + delta_bias

Predicción:

- Sumatoria de los pesos por los valores de entrada + bias

### Inferencia:

- Este perceptron puede clasificar instancias simultaneamente en 3 diferentes clases, cual lo hace un clasificador multietiqueta, Tambien puede ser usado para la clasificacion multiclase (Ejemplo anterior).

- Con aplicación de algebra lineal se puede calcular la salida de multiples instancias en una sola capa.
  $h_{W, b}(X) = f(XW+b)$
  - $X$ representa la matriz de entrada de rasgos (filas por instancia y columnas por rasgos)
  - $W$ el peso de la matriz (sus conecciones)(filas por entrada, y columna por neurona).
  - $b$ el sesgo o bias.
  - $f$ la función de activación aplicada al perceptron.

### Perceptron Learning Rules:

$w_{i,j}^{\text{(next step)}} = w_{i,j} + η(y_j − \bar{y_j})x_j$
$b_{ij}^{(\text{next step})} = b_{ij}+ η(y_j − \bar{y_j})$

- $w$ es el peso entre las conecciones (i entrada, j neurona).

- $x$ es el valor de entrada actual de la instancia

- $y_j$ es el valor de salida para la instancia actual.

- $\bar{y_j}$ es el valor de salida de la neurona para la instancia actual.

- $η$ es el Learning Rate (Ratio de aprendizaje).

La frontera de decisión de cada neurona de salida es linear, entonces el perceptrón es incapaz de aprender patrones complejos. Si las instancias de entrenamiento son linealmente separables. Rosenblatt demostró que este algoritmo podría converger a una solución. Esto lo llamó teorema de convergencia del perceptrón.

> [!info] Perceptron Convergence Theorem in Neural Networks - GeeksforGeeks  
> Your All-in-One Learning Portal: GeeksforGeeks is a comprehensive educational platform that empowers learners across domains-spanning computer science and programming, school education, upskilling, commerce, software tools, competitive exams, and more.  
> [https://www.geeksforgeeks.org/deep-learning/perceptron-convergence-theorem-in-neural-networks/](https://www.geeksforgeeks.org/deep-learning/perceptron-convergence-theorem-in-neural-networks/)

#### El problema del XOR

Marvin Minsky y Seymour Papert en 1969 enumeraron un conjunto de serios problemas que presentaba el perceptrón, en particular la incapacidad de resolver el problema lógico del XOR o problemas triviales basados en XOR.

![](./assets//image%203%204.png)

---

# The Multilayer Perceptron

Ante las limitaciones del perceptrón como unidad. El resultado de Llamarse ANN es por las MLP (Multilayer perceptron), Una MLP puede resolver problemas de XOR debido a la combinación de salidas en varias capas, asi se complejizarían del modelo generado.

Un MLP esta compuesto por una capa de entrada, una o más capas de TLUs llamadas capas ocultas, y una capa final de TLUs, llamada capa de salida. Las capas cercanas a la capa de entrada se denominan comúnmente “lower layers”, y las capas cercanas a la de salida “upper layers”

![](./assets/image%204%203.png)

- _Se le denomina Deep Neural Network a las ANN que contienen una alta profundidad en las capas ocultas, aunque con la actualidad se aumenta cada ves mas esta concepción._

### Entrenamiento del Modelo (Teórico)

El algoritmo de entrenamiento del perceptron propuesto por Rosenblatt estuvo inspirado en el postulado de aprendizaje de Hebb.

En su libro ‘The Organization of Behavior’ publicado en 1949, Hebb sugiere que cuando una neurona biológica a menudo excita a otra neurona, entonces la conexión entre ambas se fortalece, este planteamiento ahora es conocido como la regla de Hebb o el aprendizaje Hebbiano.

Hebb propuso esta concepción como la base del aprendizaje asociativo, a nivel celular, resultando en una modificación que fortaleció el patrón de actividad para el caso de un conjunto de células nerviosas espacialmente distribuidas. Esta afirmación se hizo desde una perspectiva neurobiológica

## Backpropagation:

La red retropropagada trabaja bajo aprendizaje supervisado y por tanto necesita un conjunto de entrenamiento que le describa cada salida y su valor de salida esperado de la siguiente forma:

- $(x_1,y_2),(x_1,y_2),...,(x_n,y_n)$
  Donde $x_n$ es una entrada a la red y $y_n$ es la correspondiente salida deseada para el patrón enésimo. El algoritmo debe ajustar los parámetros de la red para minimizar el Error Cuadrático Medio (MES).
  $$MSE = \frac{1}{n}\sum^n_{i = 1}(y - \bar{y})^2$$

### Flujo positivo (FeedForward) ->

1. Cada una de las neuronas de la capa oculta tiene como salida $a^n_j$, dada por:
   - $a^0_j = f^0(\sum^q_{i = 1}(w^0_{ji}*p_i) + b^0_j)$
2. $f^n$: Función de transferencia de las neuronas de la capa oculta. Las salidas de $a_j^n$ de las neuronas de la capa oculta (m componentes) son las entradas a los pesos de conexión de la capa de salida, este comportamiento se describe por:
   - $n_j^s = \sum^m_j=1(w^s_{kj}*a^0_j) + b^s_k$
3. La red produce una salida final:
   - $\bar y^s_k = f^s(n^s_k)$
4. Se calcula el error obtenido comparando la $\bar y^s_k$ con $y_k$ y se promedia al **MSE**

### Flujo Negativo (Backward)

![](./assets/image%205%203.png)

Partimos del **MSE** anterior como punto de partida para iniciar el proceso de retro propagación

La capa de salida es quien se enfrenta primeramente a la salida global de la red, por tanto representa la capa de análisis inicial:

$$\frac{\partial C}{\partial w} = δ^L * \frac{\partial z^L}{\partial w^L}$$
donde:

$\delta ^ i$ representa el error imputado a la neurona.

El error imputado a las neuronas de la última capa $\delta ^L$ representa:

$$\delta ^L = \frac{\partial C}{\partial w} * \frac{\partial z^L}{\partial w^L}$$

- Para cada capa anterior, el delta se calcula así:

  $$δ^l = [(W^{(l+1)})^Tδ^{(l+1)}] * f'(z^{(l)}$$
  - Tomas el error imputado a la capa siguiente.
  - Lo multiplicas por los pesos de esa capa siguiente **transpuestos** para poder hacer el producto entre matrices.
  - Lo ponderas por la derivada de la activación de la capa actual

- Asignación de pesos:
  Capa de salida:
  - $\frac{\partial{C}}{\partial{b^L}} = δ^{L}$
  - $\frac{\partial{C}}{\partial{w^L}} = \alpha * (a^{l - 1}(z^{l-1}) δ^{L})$
    Capas intermedias:
  - $\frac{\partial{C}}{\partial{b^{l-1}}} = δ^{l}$
  - $\frac{\partial{C}}{\partial{w^l}} = \alpha * (a^{l - 1}(z^{l-1})' \space δ^{L})$
    **Nota:** Para la capa de entrada, los datos de entrada que se multiplican por los deltas son los datos de entrada al modelo.

## Arquitectura 1: MLP Orientado a Objetos (SGD Puro)

**Ubicación:** `src/mlp_sgd/`

Esta primera aproximación busca modelar la red neuronal tal cual la imaginamos de forma biológica y en los diagramas de libros de texto: como una colección de unidades individuales conectadas entre sí.

### Estructura de Clases:

1. **`SinglePerceptron`:** Representa una única neurona matemática. Tiene su propio vector de pesos (`weights`), un `bias`, y una función de activación individual.
2. **`Layer`:** Agrupa en un bloque (`list()`) múltiples objetos `SinglePerceptron`. Delega las tareas iterando sobre cada uno de ellos.
3. **`SgdMLP`:** Controla la red completa y la propagación.

### Método de Entrenamiento:

Utiliza un **Stochastic Gradient Descent (SGD)** completamente puro. Toma **una sola muestra `(x, y)`** a la vez, hace el _forward pass_, calcula el error y aplica el _backward pass_ actualizando los pesos de la red inmediatamente antes de procesar el siguiente dato.

- Es el enfoque más visual y directo para entender cómo "aprende" cada neurona individual. La lógica de propagación es sumamente explícita.

Pero sucede que:

- Al depender de bucles `for` nativos de Python para iterar neurona por neurona y muestra a muestra, el código pierde toda la capacidad de procesamiento matemático del procesador. Es muy ineficiente y no escalará a datasets medianos o grandes.
- Ademas de que aplica la regla de aprendizaje directamente sobre cada neurona unitariamente siendo afectadas cada una por errores no producidos por si misma.

## Arquitectura 2: MLP Enfoque Matricial

**Ubicación:** `src/mlp/`

Al entender las limitaciones teóricas del primer enfoque, esta segunda iteración elimina el concepto aisaldo de la neurona y adopta el cálculo matricial. Es el mismo enfoque base que utilizan librerías de la industria como **Keras**.

### Estructura de Clases:

1. **`Layer`:** Ahora representa una **matriz de pesos** completa bidimensional ($W$) y un vector de sesgos ($b$). En lugar de iterar, procesa todos los datos simultáneamente. Además actúa como una "caché", guardando de forma inteligente lo que recibe (`input_x`) y lo que emite (`output_z`) en el pase hacia adelante para consultarlo más tarde.
2. **`MLP`:** Gestiona la orquestación algebraica.

### Método de Entrenamiento y Backpropagation:

Usa **Mini-Batch Gradient Descent**. En lugar de pasar un solo dato, empaqueta conjuntos de ejemplos (matrices multivariables) definidos por `batch_size`.
El avance (`forward(X)`) se resume de forma más rápida en álgebra lineal:
$$ Z^{(l)} = X \cdot W^{(l)} + b^{(l)} $$
$$ A^{(l)} = f(Z^{(l)}) $$

El retroceso (`backward(X, y)`) calcula los gradientes aplicando la **Regla de la Cadena**. Transmite el error ($\delta$) aplicando la derivada de la función de activación de la capa _actual_ (ej. `leaky_relu` o `sigmoid`) y usa combinaciones de matrices transpuestas `.T` para alinear las dimensiones de forma natural hacia atrás.

- Estabilidad de descenso en la curva de pérdida (MSE) al promediar los gradientes del lote (_batch_).

# Conclusiones

**Este proyecto nace de la idea de comprender el aprendizaje automático y el aprendizaje profundo**. Representó para mi una herramienta y a la vez un ejercicio mental para mi aprendizaje como estudiante

## Referencias:

Plataforma **DataCamp**:

- [Forward Propagation Neural Networks](https://www.datacamp.com/es/tutorial/forward-propagation-neural-networks)
- [Mastering Backpropagation](https://www.datacamp.com/tutorial/mastering-backpropagation)
- [Perceptrones multicapa en el aprendizaje automático: Guía completa](https://www.datacamp.com/es/tutorial/multilayer-perceptrons-in-machine-learning)

**YouTube**:

- [Neural Networks by 3Blue1Brown ](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Las Matemáticas detrás del Backpropagation by DotCSV](https://youtu.be/M5QHwkkHgAA)
- [Build Neural Network from Scratch by Samson Zhang](https://youtu.be/w8yWXqWQYmU)

**Literatura:**

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Introduction to Machine Learning, fourth edition](https://books.google.com/books/about/Introduction_to_Machine_Learning_fourth.html?id=tZnSDwAAQBAJ)

**Otras:**

- Asigntura de "Procesamiento de Señales Digitales II".

---

### 🔧 Instalación:

Sigue estos pasos para clonar el repositorio, instalar las dependencias necesarias y ejecutar los notebooks Jupyter.

Abre tu terminal y ejecuta:

```bash
git clone https://github.com/rgz-Cristian/AI-ML-Personal-Learning-Project.git
```

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate

# En macOS / Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

Nota: El objetivo principal es la implementación de los MLPs mas que el uso que le doy, los .ipynb son meras pruebas de capacidad de mis modelos.

---
