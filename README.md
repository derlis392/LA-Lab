# LA-Lab v0.6.4 — Laboratorio Virtual de Analítica del Aprendizaje

Aplicación educativa abierta con siete módulos y **14 prácticas guiadas basadas en dos colecciones reales de UCI**. Las prácticas son preguntas diferentes sobre los mismos conjuntos de datos, no 14 datasets independientes.

## Instalación

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cambios de v0.6.4

- **Caso 12: Respuestas y dificultades en matemáticas**: MathE, UCI 1031 (9546 respuestas, 8 variables). Se estudian respuestas correctas/incorrectas por tema y dificultad; cada registro es una respuesta, no un estudiante independiente.
- **Caso 13: Hábitos de estudio y desempeño universitario**: Higher Education Students Performance Evaluation, UCI 856 (145 estudiantes, 31 características y calificación objetivo). Los campos codificados se interpretan con el diccionario oficial.
- **Caso 14: Factores de permanencia universitaria**: Predict Students’ Dropout and Academic Success, UCI 697. Se analizan matrícula, becas y trayectoria, sin inferencia causal.
- Los tres casos tienen **fuentes originales diferentes** y se cargan automáticamente, con descarga del CSV de la práctica y enlace a fuente/artículo.
- El límite global de carga sigue siendo **10 MB**. No se necesita subir ZIP para casos guiados.
- `app.py`, `reporting.py`, `requirements.txt`, `README.md` y `.streamlit/config.toml` están sincronizados con v0.6.4.
- PDF y XLSX se mantienen en el módulo 6. Los logotipos proporcionados se conservan en `assets/`.

## Fuentes académicas y acceso directo

- [MathE — UCI 1031](https://archive.ics.uci.edu/dataset/1031/dataset+for+assessing+mathematics+learning+in+higher+educa) — [artículo](https://doi.org/10.1016/j.dib.2024.110236)
- [Higher Education Students Performance Evaluation — UCI 856](https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation) — [registro y DOI](https://doi.org/10.24432/C51G82)
- [Predict Students' Dropout and Academic Success — UCI 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) — [artículo](https://doi.org/10.3390/data7110146)
- [Student Performance — UCI 320](https://archive.ics.uci.edu/dataset/320/student+performance)
- [OULAD — referencia complementaria](https://analyse.kmi.open.ac.uk/open_dataset)

## Alcance y cautelas

Las prácticas 12–14 **no miden clics en entornos virtuales**: se redefinieron para analizar respuestas matemáticas, desempeño y permanencia a partir de tres colecciones UCI diferentes. La aplicación descarga las fuentes originales al ejecutarse, por lo que necesita conectividad desde Streamlit Cloud. Los modelos y comparaciones son exploratorios, no se replican automáticamente los resultados de los artículos y no deben usarse para decisiones individuales automatizadas.

## Despliegue

Subir **todos los archivos de este ZIP** a la raíz de `main` en GitHub, conservando `.streamlit/config.toml` y `assets/`. Si Streamlit Cloud mantiene una sesión antigua, usar `Manage app → Reboot app`. Confirmar `v0.6.4` en la barra lateral y en un PDF recién generado.

## Comprobaciones sugeridas

1. Abrir los casos 12, 13 y 14 y confirmar que se carga la tabla sin solicitar archivos.
2. Descargar el CSV de cada caso y revisar sus variables.
3. Ejecutar una regresión, clasificación y agrupamiento, según el objetivo seleccionado.
4. Generar PDF y XLSX desde el módulo 6.
5. Verificar que el laboratorio libre mantiene 10 MB de carga.


## MathE (caso 12)

El archivo `data/mathe_original.csv` procede del ZIP original de MathE aportado por el usuario. Contiene **9,546 respuestas y 8 variables**. Se ha convertido desde Windows-1252, con separador `;`, a CSV UTF-8 con separador coma, sin modificar las observaciones. El caso 12 lo lee directamente del repositorio, sin llamadas de red ni carga manual. Se conserva el enlace a la ficha UCI y al artículo científico. Para desplegar, es imprescindible subir la carpeta `data/`. Los casos 13 y 14 siguen usando `ucimlrepo`.

