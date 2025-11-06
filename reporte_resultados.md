# P5_IA: Clasificador de Distancia Mínima Aplicado a Enfermedades Cardíacas

## 1. 🎯 Introducción

El presente informe evalúa la eficacia de un **Clasificador de Distancia Mínima** implementado desde cero para predecir la presencia de enfermedades cardíacas utilizando el dataset **Heart Disease UCI (`heart.csv`)**.

---

## 2. 🛠️ Metodología

### 2.1. Preprocesamiento (Ajuste Necesario)
Para el correcto funcionamiento del clasificador, se aplicó **One-Hot Encoding** a las variables categóricas (`sex`, `chestpaintype`, `st_slope`, etc.) y **Estandarización (`StandardScaler`)** a todas las características numéricas.

### 2.2. Validación y Métricas
* **Métodos:** Hold-Out 70/30 y 10-Fold Cross-Validation.
* **Métricas:** Accuracy y Matriz de Confusión.

---

## 3. 📊 Resultados Obtenidos (Datos de Ejecución Final)

| Métrica / Validación | Hold-Out 70/30 | 10-Fold Cross-Validation |
| :--- | :--- | :--- |
| **Accuracy** | **0.8804** | **0.8551** (Promedio) |
| **Desviación Estándar** | N/A | $\pm$ **0.0343** |

### 3.1. Matriz de Confusión - Hold-Out 70/30
$$
\begin{pmatrix}
\text{111 (VN)} & \text{12 (FP)} \\
\text{21 (FN)} & \text{132 (VP)}
\end{pmatrix}
$$

### 3.2. Matriz de Confusión Total - 10-Fold Cross-Validation
$$
\begin{pmatrix}
\text{344} & \text{66} \\
\text{67} & \text{441}
\end{pmatrix}
$$

---

## 4. 📈 Análisis y Discusión

### 4.1. Desempeño y Fiabilidad
El clasificador de Distancia Mínima logró un **sólido Accuracy promedio del 85.51%**, demostrando que es viable para este problema. La **baja desviación estándar ($\pm 0.0343$)** confirma que el modelo es **consistente y robusto** a través de las diferentes particiones de los datos.

### 4.2. Evaluación de Errores (Falsos Negativos)
Aunque el rendimiento es alto, la matriz de confusión revela que el modelo produce **más Falsos Negativos (21 FN)** que Falsos Positivos (12 FP) en el conjunto de prueba. Esto indica que el modelo es ligeramente más propenso a decir que un paciente está sano cuando en realidad está enfermo. Este es un error crítico que debe minimizarse en futuras iteraciones.

### 4.3. Recomendaciones
Para minimizar el error crítico (FN), se sugiere:
1.  **Explorar Clasificadores No Lineales:** Evaluar modelos como *Support Vector Machines* o *Random Forest*.
2.  **Ajuste de Umbrales:** Si la tarea es solo minimizar Falsos Negativos, se puede ajustar el umbral de decisión del clasificador.

---

## 5. 🚀 Repositorio y Prueba del Código

### Prueba de Ejecución
La demostración del código se realiza mediante la siguiente imagen, que valida la ejecución exitosa del script `min_distance_classifier.py` y la obtención de los resultados reportados en la consola.

![Resultados Finales del Clasificador](resultados_consola.png)
