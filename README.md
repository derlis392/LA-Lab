# LA-Lab v0.5 — Laboratorio Virtual de Analítica del Aprendizaje

Actualización sobre LA-Lab v0.4. Aplicación educativa en Streamlit con 7 módulos, 14 prácticas sobre fuentes académicas, visualización, regresión, clasificación, agrupamiento e informes académicos enriquecidos.

## Cambios principales

- **Informe PDF de varias páginas** con descripción del caso, procedencia, estadística descriptiva, matriz de correlación, gráfico exploratorio (si se ejecutó), figuras específicas del experimento, métricas, comparación de algoritmos, interpretación contextualizada, recomendaciones y limitaciones.
- **Excel ampliado**: datos, metadatos, estadísticas, correlaciones, experimento, métricas, comparación, predicciones, clasificación por clase o agrupamiento según la tarea ejecutada.
- **Interpretaciones más útiles** para clasificación desbalanceada, regresión frente a baseline y agrupamiento con silhouette, incluidas en el laboratorio y el informe.
- **Comparación gráfica de algoritmos** (F1 macro o MAE según la tarea), recomendación cautelosa de k exploratorio.
- **Persistencia en sesión** de resultados y visualizaciones. Cambiar la fuente o limpiar los datos invalida los experimentos anteriores.
- **Enlace de artículo de Student Performance** dirigido a la ficha bibliográfica de UCI, en lugar de una búsqueda de OpenML no equivalente a la publicación.

## Instalación

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Actualizar Streamlit Community Cloud

Copiar a la raíz del repositorio `app.py`, `reporting.py` y `requirements.txt` (o subir el contenido completo del ZIP conservando `.streamlit/config.toml`). El nuevo archivo **`reporting.py` es obligatorio**: sin él, `app.py` no arrancará. La aplicación debería redesplegarse al actualizar la rama conectada.

## Uso del informe

1. Elegir un caso y ejecutar los módulos deseados.
2. En «7. Minería de datos», seleccionar algoritmo y pulsar «Ejecutar análisis».
3. Ir a «6. Informe» y descargar el PDF y Excel.
4. El informe **solo** incorpora los experimentos ejecutados en la sesión actual. Para probar diferentes modelos y compararlos, activar «Comparar todos los algoritmos» antes de ejecutar.

## Límites conocidos

- Las **14 prácticas no representan 14 datasets independientes**. Comparten varias fuentes. OULAD requiere preparación manual de las tablas originales y se enlaza a la fuente oficial.
- La comparación de modelos sobre una sola partición es **exploratoria**; para seleccionar un modelo con rigor, se requiere validación dentro del entrenamiento y prueba final independiente.
- El PDF se genera con gráficos reproducidos a partir de los resultados numéricos almacenados; no exporta capturas exactas de Plotly.
- Los indicadores automáticos no acreditan causalidad, perfiles pedagógicos validados ni pertinencia para decisiones automatizadas sobre estudiantes.
- Los logotipos oficiales no están incluidos: requieren archivos autorizados.

## Fuentes

- Student Performance: https://archive.ics.uci.edu/dataset/320/student+performance
- Predict Students' Dropout and Academic Success: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- OULAD: https://analyse.kmi.open.ac.uk/open_dataset


## v0.5.2 — correcciones consolidadas
- Cabecera: logotipos UNAM/CUAED en columnas con proporciones diferenciadas y sin etiquetas redundantes.
- Reportes: versión unificada v0.5.2 en app, PDF y pie de página; imágenes más compactas; prevención de histogramas duplicados; leyendas específicas y referencias sin URLs repetidas.
- Validación pendiente tras despliegue: probar PDF nuevo en Streamlit Cloud, las tres tareas de ML y exportación Excel.
- Los resultados de comparación entre modelos en prueba son exploratorios: no elegir el modelo final optimizando sobre ese conjunto.


## v0.5.3 — Corrección de despliegue y presentación
- Logotipos desplazados 52 px hacia abajo en el encabezado de Streamlit; tamaños explícitos.
- Encabezado de «Fuentes y reproducibilidad» unido a su contenido para evitar título huérfano.
- Identificador v0.5.3 en aplicación y PDF.
- Para desplegar: subir **app.py y reporting.py**, además de assets/unam.png y assets/cuaed.png, en la raíz correcta del repositorio; comprobar que Streamlit usa la rama y el archivo app.py actualizados.
- Verificación rápida: la interfaz y un PDF nuevo deben mostrar ambos v0.5.3. Si no, se está ejecutando código anterior.


## Control de versiones de informes (v0.5.4)
Actualizar **app.py y reporting.py juntos**, en la raíz del repositorio. La pantalla del módulo 6 muestra la versión real del generador importado. Si no es 0.5.4, bloquea la descarga del PDF en lugar de ofrecer una versión anterior. Descargar `informe_la_lab_v054.pdf` y verificar que título y pie muestren v0.5.4. La sección 7 comienza en página nueva para evitar encabezados huérfanos.

## v0.6 — Casos OULAD 12, 13 y 14

**Cambios:** se integró `oulad.py`, que prepara directamente las tablas originales del ZIP de OULAD en la modalidad **Casos guiados**. No se distribuyen ni se generan registros ficticios. La carga es explícita, ya que OULAD no es un único CSV y el servidor público no incorpora el dataset por defecto.

1. Descarga el ZIP oficial de OULAD desde la ficha científica del caso.
2. Selecciona el caso 12, 13 o 14 y carga el ZIP mediante el control que aparece en el panel principal (límite 200 MB).
3. Pulsa **Preparar datos reales del caso OULAD**. Los casos 12, 13 y 14 integran respectivamente `studentInfo + studentVle`, `studentInfo + studentAssessment + assessments` y `studentInfo + studentRegistration`.
4. Continúa por los módulos habituales o descarga la tabla integrada para verificar las uniones.

El caso 12 agrega `sum_click` por estudiante, módulo y presentación en bloques de 200 000 filas, evitando cargar `studentVle.csv` completo en memoria. Los casos 13 y 14 agregan evaluaciones o fechas de registro/desregistro. Se conservan los identificadores y el resultado académico para análisis exploratorio; **no deben usarse identificadores como predictores** ni el resultado como predictor de sí mismo. Las ausencias de registros de actividad se codifican como cero clics registrados; esto no acredita ausencia de aprendizaje. La variable `registro_retirado` se basa en la presencia de fecha de baja.

**Archivos que deben actualizarse juntos:** `app.py`, `reporting.py`, `oulad.py`, `requirements.txt`, `.streamlit/config.toml`, `assets/` y `README.md`. El PDF y la interfaz deben indicar `v0.6`. El límite de subida del servidor se amplió a 200 MB para admitir el ZIP original; se recomienda no subir datos personales ni confidenciales.

**Limitaciones:** la aplicación necesita que el usuario descargue el ZIP oficial; no se incluye el corpus original en el repositorio. La preparación OULAD depende de los nombres de archivo y columnas del esquema oficial. Las integraciones deben comprobarse con un ZIP OULAD real después del despliegue; las pruebas locales de estructuras no sustituyen esa verificación.
