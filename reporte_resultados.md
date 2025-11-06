# 📄 Reporte: Clasificador de Distancia Mínima en Heart Failure Dataset (P5_IA)

## 1. 🎯 Introducción

El presente informe detalla la implementación y evaluación de un **Clasificador de Distancia Mínima** (Nearest Centroid) aplicado al conjunto de datos **Heart Failure Clinical Records** (Kaggle). El objetivo fue clasificar la ocurrencia de un evento de muerte (`DEATH_EVENT`).

El desempeño se evaluó con la **Exactitud (Accuracy)** y la **Matriz de Confusión**, utilizando los métodos de validación **Hold-Out 70/30** y **10-Fold Cross-Validation**.

---

## 2. 🛠️ Metodología

### 2.1. Dataset y Preprocesamiento
* **Dataset:** Heart Failure Clinical Records (299 instancias).
* **Preprocesamiento:** Se aplicó **Estandarización (StandardScaler)** a todas las características. Este paso es fundamental, ya que el clasificador se basa en la Distancia Euclidiana, sensible a las diferencias de escala entre las variables.

### 2.2. Clasificador Implementado
Se utilizó un clasificador customizado que:
1.  Calcula el **centroide** (media) de las características para la clase 0 y la clase 1 en el conjunto de entrenamiento.
2.  Asigna una nueva instancia a la clase cuyo centroide es el **más cercano** (Distancia Euclidiana mínima).

### 2.3. Validación y Métricas
* **Métodos de Validación:** Hold-Out 70% entrenamiento / 30% prueba (estratificado) y 10-Fold Cross-Validation (mezclado).
* **Métricas:** Accuracy, Matriz de Confusión, y Desviación Estándar (para CV).

---

## 3. 📊 Resultados Obtenidos

| Métrica / Validación | Hold-Out 70/30 | 10-Fold Cross-Validation |
| :--- | :--- | :--- |
| **Accuracy** | **0.6999** | **0.6756** (Promedio) |
| **Desviación Estándar** | N/A | $\pm$ **0.0617** |

### 3.1. Matriz de Confusión - Hold-Out 70/30

Esta matriz resume el desempeño en el 30% de los datos de prueba:

| Real $\setminus$ Predicho | Clase 0 (No Murió) | Clase 1 (Murió) |
| :---: | :---: | :---: |
| **Clase 0 (No Murió)** | **59** (Verdaderos Negativos) | 3 (Falsos Positivos) |
| **Clase 1 (Murió)** | **24** (Falsos Negativos) | **4** (Verdaderos Positivos) |

**Formato Matricial:**
$$
\begin{pmatrix}
59 & 3 \\
24 & 4
\end{pmatrix}
$$

### 3.2. Matriz de Confusión Total - 10-Fold Cross-Validation

Esta matriz es la suma acumulada de los resultados obtenidos en los 10 tests de validación, representando la clasificación de las 299 muestras.

| Real $\setminus$ Predicho | Clase 0 (No Murió) | Clase 1 (Murió) |
| :---: | :---: | :---: |
| **Clase 0 (No Murió)** | **185** | 18 |
| **Clase 1 (Murió)** | **79** | **17** |

**Formato Matricial:**
$$
\begin{pmatrix}
185 & 18 \\
79 & 17
\end{pmatrix}
$$

---

## 4. 📈 Análisis y Conclusiones

### 4.1. Robustez del Modelo
El resultado del **10-Fold CV (67.56%)** es la estimación más fiable del rendimiento del clasificador. La baja **desviación estándar ($\pm 0.0617$)** indica que el modelo es relativamente **consistente** y robusto ante las diferentes particiones de los datos.

### 4.2. Problema de Sesgo (Falsos Negativos)
A pesar de un $Accuracy$ que parece aceptable (~67%), el modelo presenta un sesgo crítico:
* El clasificador es muy bueno prediciendo la **Clase 0 (Sobrevive)**.
* Sin embargo, en la matriz Hold-Out, produjo **24 Falsos Negativos (FN)** frente a solo **4 Verdaderos Positivos (VP)** para la Clase 1. Esto significa que **falló en detectar al 85.7%** ($24/(24+4)$) de los pacientes que murieron.
* Clínicamente, un modelo con alta tasa de FN es inaceptable, pues minimiza la identificación del riesgo. La simple Distancia Mínima no es suficiente para distinguir el centroide de la Clase 1 (Muerte) de la Clase 0 (Sobrevive).

### 4.3. Recomendaciones
El Clasificador de Distancia Mínima es demasiado simple para este problema. Para mejorar el rendimiento, se sugiere:
1.  **Explorar Modelos Más Flexibles:** Utilizar clasificadores no lineales como *Support Vector Machines* o *Random Forest*.
2.  **Mitigar Desbalance de Clases:** Aplicar técnicas como el sobremuestreo (e.g., SMOTE) o el ajuste de pesos de clase, ya que la clase "Muerte" es minoritaria.

---

## 5. 🔗 Repositorio y Prueba del Código

El código fuente se encuentra en el archivo `min_distance_classifier.py` dentro del repositorio.

**Prueba de Ejecución:**
La demostración se realiza al ejecutar el script, que imprime los resultados de las métricas en la consola:

```bash
python min_distance_classifier.py