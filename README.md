# Predicción de Fuga de Talentos

**Autores:** Eduardo Estefanía Ovejero (eduardo.estefania@cunef.edu), David Blasco Tena (david.blasco@cunef.edu) y Alejandro Jesús Castro García (alejandro.castro@cunef.edu)

## Índice del Proyecto

1. **01_eda_exploration**
2. **02_feature_engineering**
3. **03_modelado_desbalanceado**
    1. **03.1_modelado_undersampling**
4. **04_modelado(opcional)**

## Resumen Proyecto

Este proyecto aborda un desafío crítico y de alto impacto financiero en el ámbito de los Recursos Humanos: **la fuga de talento**. El objetivo principal es predecir qué empleados tienen un alto riesgo de abandonar la empresa, formulando este escenario como un problema de **clasificación binaria** mediante técnicas de Aprendizaje Automático (Machine Learning).

El reto técnico central de este conjunto de datos radica en el **fuerte desbalanceo de clases**: la gran mayoría de los empleados (clase mayoritaria) permanece en la compañía, mientras que solo una fracción minoritaria (~16%) decide marcharse.

Para comprender a fondo la matemática subyacente de los algoritmos y abordar este problema de desbalanceo paso a paso, el proyecto se ha estructurado en tres iteraciones de modelado evolutivas:

1. **Modelo Base Analítico (Sin balancear):** Se implementó un algoritmo de Regresión Logística "desde cero" utilizando exclusivamente cálculo matricial, derivadas y descenso de gradiente con `numpy`. Este modelo sirve como línea base y demuestra el sesgo natural que sufren los algoritmos hacia la clase mayoritaria cuando no se intervienen los datos, generando un modelo conservador de baja Exhaustividad (Recall).

2. **Modelo Analítico Balanceado (Random Undersampling):**
   Manteniendo la implementación matemática manual del algoritmo lineal, se introdujo una técnica de submuestreo aleatorio para igualar el volumen de ambas clases en el entrenamiento. Este enfoque demuestra la corrección del sesgo, logrando un modelo preventivo capaz de maximizar la detección de fugas (alto Recall), comportamiento óptimo y deseado para las políticas de retención de RRHH.

3. **Modelo Opcional (Scikit-Learn + SMOTE):**
   Como marco de validación, se incluye una aproximación utilizando herramientas de alto nivel de la industria. Mediante el uso de `Pipelines`, generación de datos sintéticos (SMOTE) y optimización de hiperparámetros (`GridSearchCV`), este modelo establece el límite de rendimiento (*techo*) predecible para este conjunto de datos.

A través de este proceso iterativo, el proyecto abarca el ciclo de vida completo de los datos: desde el Análisis Exploratorio (EDA) y la ingeniería de características, hasta la programación analítica del algoritmo y la traducción de métricas técnicas avanzadas (ROC AUC, Balanced Accuracy) en impacto y decisiones de negocio.

---

# 1. Descripción del Problema
La retención del talento humano es uno de los mayores desafíos para los departamentos de Recursos Humanos en la actualidad. Cuando un empleado abandona una empresa (lo que se conoce como Attrition o desgaste), la organización se enfrenta a costes significativos: gastos de reclutamiento, tiempo de formación para el sustituto, pérdida de conocimiento interno y una posible disminución en la productividad del equipo.

Objetivo del proyecto: El objetivo principal de este trabajo es analizar los datos históricos de los empleados de una empresa para descubrir qué factores (salario, distancia al trabajo, horas extras, satisfacción, etc.) influyen más en la decisión de abandonar la compañía. A partir de este análisis, se desarrollará un modelo de Machine Learning capaz de predecir la probabilidad de que un empleado actual deje la empresa, permitiendo a Recursos Humanos tomar medidas preventivas.

# 2. Origen de los Datos
Dataset: IBM HR Analytics Employee Attrition & Performance

Enlace de obtención: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Evaluación del dataset:

Ventajas: Es un conjunto de datos muy completo (35 variables y 1470 registros) que incluye una gran variedad de atributos demográficos, económicos y de bienestar laboral. Está estructurado de forma muy limpia, lo que facilita la aplicación directa de técnicas de Machine Learning y el Análisis Exploratorio (EDA).

Desventajas: Se trata de un dataset generado sintéticamente por científicos de datos de IBM (no son datos reales extraídos de una base de datos corporativa real, aunque imitan muy bien el comportamiento real). Además, la variable objetivo (Attrition) es binaria (Sí/No) y agrupa todos los motivos de salida (jubilación, despido, renuncia voluntaria), lo que impide analizar las causas específicas de cada tipo de baja.

# 3. Planteamiento Concreto y Enfoque Técnico
Dado que el objetivo es predecir si un empleado abandona la empresa o no, nos encontramos ante un problema de Clasificación Binaria. La variable objetivo (o Target) es la columna Attrition, que toma los valores "Yes" (1) o "No" (0).

Para resolver este problema, se aplicará un modelo de Regresión Logística, que es el algoritmo idóneo para modelar la probabilidad de que ocurra un evento binario. El ajuste de los pesos (parámetros) del modelo se realizará mediante el algoritmo de optimización de Descenso de Gradiente, minimizando la función de coste logística.

## 3.1. Enfoque Algorítmico: Regresión Logística Analítica
Siguiendo las directrices académicas de la asignatura, el núcleo del modelado no se basa en librerías de alto nivel, sino en una **implementación analítica desde cero** utilizando `numpy`. El enfoque técnico se desglosa en:

1.  **Función de Activación (Sigmoide):** Se utiliza para mapear la salida lineal de las 39 variables ($z = X \cdot w + b$) en un rango probabilístico entre 0 y 1.
2.  **Función de Coste (Binary Cross Entropy - BCE):** Se ha seleccionado esta función por su capacidad para penalizar severamente las predicciones seguras pero incorrectas, lo cual es fundamental para forzar al modelo a aprender patrones en clases minoritarias.
3.  **Optimización (Descenso de Gradiente):** Se implementó un algoritmo iterativo de optimización para ajustar los pesos ($w$) y el sesgo ($b$) mediante el cálculo de gradientes de la pérdida BCE respecto a cada parámetro.
4.  **Operaciones Matriciales:** Para soportar las 39 variables del dataset de forma eficiente, se ha utilizado álgebra lineal matricial en lugar de bucles escalares, permitiendo un entrenamiento multivariable robusto.

## 3.2. Estrategia frente al Desbalanceo de Clases
El dataset presenta un sesgo significativo (aprox. 84% de permanencia vs 16% de abandono). Para mitigar el riesgo de que el modelo ignore sistemáticamente a la clase minoritaria, se plantean tres escenarios técnicos:
* **Modelo Base:** Sin tratamiento de balanceo para observar el sesgo natural del algoritmo.
* **Random Undersampling (Manual):** Balanceo mediante la reducción de la clase mayoritaria en el conjunto de entrenamiento para igualar la presencia de ambas clases.
* **SMOTE (Opcional):** Generación de muestras sintéticas mediante vecinos cercanos utilizando la librería *Scikit-Learn* para establecer un marco comparativo de rendimiento profesional.

# 4. Métricas de Evaluación
Para evaluar el rendimiento del modelo no nos basaremos únicamente en la Exactitud (Accuracy). Al explorar los datos preliminares, se observa que existe un desbalanceo de clases (aproximadamente un 84% de los empleados se quedan y un 16% se van). Por ello, se justificarán y utilizarán las siguientes métricas:

## Métrica Principal: Balanced Accuracy

**Balanced Accuracy** = (Sensibilidad + Especificidad) / 2

Promedia el rendimiento en **ambas clases**, por lo que:
- No se ve sesgada por el desbalance
- Penaliza modelos que ignoran la clase minoritaria
- Valora equitativamente la detección de ambos grupos

## Métricas Complementarias

1. **Precision (abandono):** De los que predecimos que abandonan, ¿cuántos realmente lo hacen?
   - Alta precisión → Pocas falsas alarmas
   - Importante para no desperdiciar recursos en intervenciones innecesarias

2. **Recall (abandono):** De los que realmente abandonan, ¿cuántos detectamos?
   - Alto recall → Detectamos la mayoría de casos de riesgo
   - Crítico para no perder empleados valiosos

3. **F1-Score (abandono):** Media armónica de precision y recall
   - Balance entre ambas métricas
   - Útil cuando queremos optimizar ambas simultáneamente

4. **Curva ROC y AUC (Area Under the Curve):** La curva ROC visualiza el compromiso (*trade-off*) entre la tasa de verdaderos positivos (Recall) y la tasa de falsos positivos en todos los umbrales de probabilidad posibles.
   - **AUC:** Es un valor único entre 0 y 1 que resume el rendimiento global del modelo. Un AUC de 0.5 equivale a predicciones aleatorias, mientras que valores más cercanos a 1.0 indican una excelente capacidad del modelo para distinguir correctamente entre los empleados que se quedan y los que abandonan, independientemente del umbral elegido.

# 5. Comparativas de modelos y conclusiones

## Métricas de Rendimiento

| Métrica | Modelo 0<br>*Manual sin balanceo* | Modelo 1<br>*Manual + Undersampling* | Modelo 2<br>*sklearn + SMOTE* |
|:---|:---:|:---:|:---:|
| **Balanced Accuracy** | 66.47% | **73.77%** | 69.46% |
| **Precision (abandono)** | **68.00%** | 37.08% | 44.44% |
| **Recall (abandono)** | 36.17% | **70.21%** | 51.06% |
| **Especificidad** | **96.76%** | 77.33% | 87.85% |
| **F1-Score (abandono)** | 47.22% | **48.53%** | 47.52% |
| **ROC AUC** | **0.8027** | — | 0.7896 |
| **Accuracy general** | 87.8% | 76.2% | **81.6%** |

> En negrita: mejor valor por métrica.

---

## Matrices de Confusión

**Modelo 0 — Regresión logística manual sin balanceo de clases**

```
                   Predicción
                No (0)   Sí (1)
Real No (0) |    239         8
Real Sí (1) |     30        17
```

**Modelo 1 — Regresión logística manual con undersampling**

```
                   Predicción
                No (0)   Sí (1)
Real No (0) |    191        56
Real Sí (1) |     14        33
```

**Modelo 2 — sklearn + SMOTE (imblearn)**

```
                   Predicción
                No (0)   Sí (1)
Real No (0) |    217        30
Real Sí (1) |     23        24
```

---

## Análisis por Criterio Empresarial

| Criterio | Mejor modelo | Justificación |
|:---|:---:|:---|
| Minimizar falsos negativos (no perder empleados en riesgo) | **Modelo 1** | Recall más alto: detecta el 70% de los abandonos |
| Minimizar falsas alarmas (no molestar a quien se queda) | **Modelo 0** | Especificidad del 96.76% y mayor precision |
| Equilibrio general entre clases | **Modelo 2** | Mejor balanced accuracy en validación (74.31%) y menor riesgo de sobreajuste que undersampling |
| Discriminación global (ranking) | **Modelo 0** | ROC AUC más alto: 0.8027 |

---

## Descripción de los Modelos

| | Modelo 0 | Modelo 1 | Modelo 2 |
|:---|:---|:---|:---|
| **Implementación** | Manual (NumPy) | Manual (NumPy) | scikit-learn |
| **Optimización** | Descenso de gradiente | Descenso de gradiente | L-BFGS / liblinear |
| **Balanceo de clases** | Ninguno | Undersampling | SMOTE (imblearn) |
| **Dependencias externas** | NumPy | NumPy | scikit-learn, imbalanced-learn |

---

## Conclusión

- El **Modelo 1 (undersampling)** es el más adecuado si el objetivo prioritario es **detectar el máximo número de empleados que van a abandonar**, asumiendo un mayor número de falsas alarmas.
- El **Modelo 0 (sin balanceo)** conviene cuando los **falsos positivos tienen un coste alto** (p. ej. planes de retención innecesarios) y se prefiere actuar solo con alta confianza.
- El **Modelo 2 (SMOTE)** representa el **mejor equilibrio** entre ambos objetivos y es el más robusto al no descartar datos reales (a diferencia del undersampling).