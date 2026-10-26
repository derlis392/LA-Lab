# LA-Lab v0.6.13 — Laboratorio Virtual de Analítica del Aprendizaje

Aplicación educativa abierta con siete módulos y **14 prácticas guiadas basadas en dos colecciones reales de UCI**. Las prácticas son preguntas diferentes sobre los mismos conjuntos de datos, no 14 datasets independientes.

## Instalación

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cambios de v0.6.13

- Se sustituyó el enlace de CiteSeerX por el registro institucional de Universidade do Minho para Student Performance (Matemáticas y Portugués). Se conservan los demás enlaces y funciones de v0.6.5.

## Cambios de v0.6.5

- Referencias científicas: enlace al artículo original de Cortez y Silva (2008) para Matemáticas y Portugués; en caso 13, enlace al capítulo científico con DOI 10.1007/978-3-030-35249-3_76 en lugar del DOI de UCI (que identifica al dataset).
- Emblemas: tamaños de presentación equilibrados sin alterar los archivos originales. Para nitidez vectorial real al ampliar, sustituir PNG por SVG oficiales autorizados.
- Etiquetado de enlaces de publicaciones y diferenciación explícita de DOI del dataset frente al artículo.


- **Caso 12: Respuestas y dificultades en matemáticas**: MathE, UCI 1031 (9546 respuestas, 8 variables). Se estudian respuestas correctas/incorrectas por tema y dificultad; cada registro es una respuesta, no un estudiante independiente.
- **Caso 13: Hábitos de estudio y desempeño universitario**: Higher Education Students Performance Evaluation, UCI 856 (145 estudiantes, 31 características y calificación objetivo). Los campos codificados se interpretan con el diccionario oficial.
- **Caso 14: Factores de permanencia universitaria**: Predict Students’ Dropout and Academic Success, UCI 697. Se analizan matrícula, becas y trayectoria, sin inferencia causal.
- Los tres casos tienen **fuentes originales diferentes** y se cargan automáticamente, con descarga del CSV de la práctica y enlace a fuente/artículo.
- El límite global de carga sigue siendo **10 MB**. No se necesita subir ZIP para casos guiados.
- `app.py`, `reporting.py`, `requirements.txt`, `README.md` y `.streamlit/config.toml` están sincronizados con v0.6.13.
- PDF y XLSX se mantienen en el módulo 6. Los logotipos proporcionados se conservan en `assets/`.

## Fuentes académicas y acceso directo

- [MathE — UCI 1031](https://archive.ics.uci.edu/dataset/1031/dataset+for+assessing+mathematics+learning+in+higher+educa) — [artículo](https://doi.org/10.1016/j.dib.2024.110236)
- [Higher Education Students Performance Evaluation — UCI 856](https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation) — [artículo/capítulo científico](https://doi.org/10.1007/978-3-030-35249-3_76) — [DOI del dataset](https://doi.org/10.24432/C51G82)
- [Predict Students' Dropout and Academic Success — UCI 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) — [artículo](https://doi.org/10.3390/data7110146)
- [Student Performance — UCI 320](https://archive.ics.uci.edu/dataset/320/student+performance) — [Cortez y Silva (2008), repositorio institucional](https://repositorium.uminho.pt/entities/publication/42d177d6-868b-4e6c-8a89-e1e7fa41c1d0)
- [OULAD — referencia complementaria](https://analyse.kmi.open.ac.uk/open_dataset)

## Alcance y cautelas

Las prácticas 12–14 **no miden clics en entornos virtuales**: se redefinieron para analizar respuestas matemáticas, desempeño y permanencia a partir de tres colecciones UCI diferentes. La aplicación descarga las fuentes originales al ejecutarse, por lo que necesita conectividad desde Streamlit Cloud. Los modelos y comparaciones son exploratorios, no se replican automáticamente los resultados de los artículos y no deben usarse para decisiones individuales automatizadas.

## Despliegue

Subir **todos los archivos de este ZIP** a la raíz de `main` en GitHub, conservando `.streamlit/config.toml` y `assets/`. Si Streamlit Cloud mantiene una sesión antigua, usar `Manage app → Reboot app`. Confirmar `v0.6.13` en la barra lateral y en un PDF recién generado.

## Comprobaciones sugeridas

1. Abrir los casos 12, 13 y 14 y confirmar que se carga la tabla sin solicitar archivos.
2. Descargar el CSV de cada caso y revisar sus variables.
3. Ejecutar una regresión, clasificación y agrupamiento, según el objetivo seleccionado.
4. Generar PDF y XLSX desde el módulo 6.
5. Verificar que el laboratorio libre mantiene 10 MB de carga.


## MathE (caso 12)

El archivo `data/mathe_original.csv` procede del ZIP original de MathE aportado por el usuario. Contiene **9,546 respuestas y 8 variables**. Se ha convertido desde Windows-1252, con separador `;`, a CSV UTF-8 con separador coma, sin modificar las observaciones. El caso 12 lo lee directamente del repositorio, sin llamadas de red ni carga manual. Se conserva el enlace a la ficha UCI y al artículo científico. Para desplegar, es imprescindible subir la carpeta `data/`. Los casos 13 y 14 siguen usando `ucimlrepo`.


## Identidad de la línea de investigación

Se incorpora `assets/linea_investigacion.svg` (archivo vectorial original proporcionado por el usuario). En la cabecera de LA-Lab, el panel **Ver figura de la línea de investigación** permite visualizarla sin rasterizar el archivo. Los emblemas institucionales UNAM y CUAED se mantienen sin cambios.

## Corrección de visibilidad v0.6.13
El logotipo TITDEE se muestra directamente en la cabecera, junto a UNAM y CUAED, sin desplegables. Se incrusta el SVG original como imagen vectorial en HTML, evitando las limitaciones de visualización de SVG de `st.image`. Verificar que `assets/linea_investigacion.svg` se suba a GitHub.

## v0.6.13 — alineación de identidad visual
Se sustituyeron las tres columnas de logotipos por una fila flex centrada con espacios uniformes, tamaños controlados y SVG original de la línea de investigación. Se conservaron los módulos y las referencias de v0.6.8.

## Corrección de informes v0.6.13
- Corregida la comparación entre `reporting.REPORT_VERSION` y la versión requerida por `app.py`, que bloqueaba todos los PDF.
- Nombres de archivo PDF, TXT y XLSX sincronizados.
- Pie de página actualizado.
- Validar generación de PDF en Streamlit tras desplegar.


## Cambios v0.6.13
- Menú: 6. Minería de datos, 7. Informe.
- PDF: tres logotipos, diccionario parcial, advertencia ordinal, nota de alcance y mejor paginación.
- Excel: nuevas hojas Diccionario, Reproducibilidad y Sintesis_pedagogica.
- Los códigos de variables ajenos a Student Performance requieren consultar su documentación original.

## v0.6.13 — Ajustes del PDF
- Los tres emblemas UNAM, CUAED y TITDEE son obligatorios en la portada del PDF. El SVG de TITDEE se rasteriza en memoria a alta resolución y se recorta según su transparencia.
- Ajuste de márgenes y espaciado para evitar páginas finales con escaso contenido; ampliación de la matriz de correlaciones.
- Eliminación de una reiteración del alcance del experimento.

## Corrección v0.6.13: PDF sin Cairo
El informe usa `assets/linea_investigacion_pdf.png`, rasterizado a alta resolución y recortado previamente desde el SVG original. El SVG se conserva para la interfaz web. No requiere CairoSVG ni bibliotecas del sistema Cairo en Streamlit Cloud.

## Nuevo módulo Motivación (v0.6.14)

En la barra lateral selecciona **Motivación**. Incluye un Excel **sintético** de 60 respuestas (`data/respuestas_sinteticas_60.xlsx`) generado con semilla fija y una coincidencia de cumpleaños forzada para probar el detector. No contiene respuestas reales. También acepta el XLSX o CSV descargado del formulario de Google Forms con las preguntas numeradas 1–10.

Funciones: validación de día/mes, conteo de coincidencias, probabilidad teórica de cumpleaños (365 o 366 fechas), simulación reproducible de grupos, frecuencias absolutas/relativas/acumuladas, medidas descriptivas, histogramas, probabilidades empíricas y condicionales, y dispersión estudio–promedio. El módulo no incorpora datos de la encuesta a los modelos de los otros módulos ni sustituye el informe PDF académico de los casos guiados.

Para desplegar, sustituye el contenido completo del repositorio por este ZIP, conservando `motivacion.py` y `data/respuestas_sinteticas_60.xlsx`. La dependencia `streamlit` ya figura en `requirements.txt`. No se requiere CairoSVG.


## v0.6.16 — Cierre de Motivación
Explicación formal y fórmula del cumpleaños, figura de crecimiento probabilístico, tabla PDF con frecuencia relativa acumulada, gráfica de estimaciones previas y conclusión automática en aplicación y PDF.
