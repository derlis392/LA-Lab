# LA-Lab v0.4 — Laboratorio virtual de analítica del aprendizaje

Aplicación Streamlit para prácticas de análisis exploratorio, visualización, matriz de correlación y minería de datos. Línea de investigación: **Tecnologías Inteligentes para la Transformación Digital de los Ecosistemas Educativos**.

## Instalación y despliegue

1. Reemplaza `app.py`, `requirements.txt` y `README.md` en la raíz del repositorio GitHub.
2. Conserva `.streamlit/config.toml` en la carpeta `.streamlit/`.
3. Streamlit Community Cloud desplegará la actualización desde `main`.
4. Para uso local: `pip install -r requirements.txt` y `streamlit run app.py`.

## Fuentes científicas originales (14 prácticas; 3 colecciones independientes)

- **Student Performance (UCI, CC BY 4.0)**: dos archivos originales `student-mat.csv` y `student-por.csv` para las prácticas 1–7. Fuente y descarga: https://archive.ics.uci.edu/dataset/320/student+performance ; DOI del dataset: https://doi.org/10.24432/C5TG7T . Artículo asociado: Cortez, P. y Silva, A. (2008), *Using Data Mining to Predict Secondary School Student Performance*. Los CSV se descargan desde el ZIP oficial cuando hay conectividad del servidor.
- **Predict Students' Dropout and Academic Success (UCI, CC BY 4.0)**: prácticas 8–11. Fuente y descarga: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success ; DOI: https://doi.org/10.24432/C5MC89 . Artículo: Realinho, V., Machado, J., Baptista, L. y Martins, M. (2022), *Predicting Student Dropout and Academic Success*, *Data*, 7(11), 146. https://doi.org/10.3390/data7110146 . Se descarga mediante `ucimlrepo`.
- **OULAD (CC BY 4.0)**: prácticas 12–14. Fuente y descarga de tablas originales: https://analyse.kmi.open.ac.uk/open_dataset ; artículo: Kuzilek, J., Hlosta, M. y Zdrahal, Z. (2017), *Open University Learning Analytics dataset*, *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171 . Por su estructura relacional y volumen, **requiere descargar los archivos oficiales, preparar la tabla para cada pregunta y cargarla desde Datos propios**. No se incluyen versiones inventadas ni se realizan uniones automáticas no documentadas.

## Limitaciones y reproducibilidad

- **Los 14 casos no son 14 datasets diferentes**: son 14 preguntas de investigación basadas en tres colecciones originales, una de las cuales tiene dos archivos por materia.
- Los artículos citados respaldan la procedencia o uso de las bases; los modelos del laboratorio son experimentos nuevos. No se asegura equivalencia numérica con resultados publicados.
- Entrenamiento/prueba con semilla y preprocesamiento en pipeline. Evaluación exploratoria, no validación externa ni decisiones individuales.
- Sin claves API ni pagos. En Streamlit Community Cloud, la concurrencia de 50 personas requiere prueba de carga y no está garantizada.
- Se admite carga de CSV, TSV, TXT delimitado, Excel, JSON de registros y Parquet hasta 10 MB. Para los CSV de Student Performance, usa `;` como separador (detección automática).
- Logotipos oficiales: colocar `assets/unam.png` y `assets/cuaed.png` con autorización institucional; no se incluyen emblemas recreados.

## Cambios en v0.4

- Clasificación: matriz de confusión en mapa de calor, métricas por clase, baseline mayoritario y advertencia de desequilibrio.
- Regresión: gráfico real vs. predicho con línea ideal, residuos y baseline de media.
- Agrupamiento: dispersión PCA coloreada por grupos, caracterización de perfiles, diagramas de caja y comparación opcional de silhouette por k.
- Comparación opcional de algoritmos y validación cruzada opcional en entrenamiento (sin ajustar modelos con el conjunto de prueba).
- Descargas adicionales PDF textual y Excel de resultados tabulares; las figuras interactivas no se incrustan en PDF.

**Alcance:** actualización de funciones; todavía no constituye reproducción verificada de resultados publicados. Los casos OULAD siguen requiriendo preparación manual. Para comparar modelos de forma científica, usar validación cruzada y un conjunto de prueba reservado para evaluación final, no seleccionar el mejor modelo por la misma prueba.
