# LA-Lab v0.6.1 — Laboratorio Virtual de Analítica del Aprendizaje

Aplicación educativa abierta con siete módulos y **14 prácticas guiadas basadas en dos colecciones reales de UCI**. Las prácticas son preguntas diferentes sobre los mismos conjuntos de datos, no 14 datasets independientes.

## Instalación

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cambios de v0.6.1

- Los casos **12, 13 y 14** ya no solicitan el ZIP grande de OULAD. Se cargan automáticamente desde UCI, con botón de descarga del CSV y pregunta orientadora específica.
- Caso 12: **Progreso curricular y resultados** (unidades curriculares del primer semestre y trayectoria).
- Caso 13: **Desempeño entre semestres** (evaluaciones y calificaciones del primer y segundo semestre).
- Caso 14: **Factores de permanencia** (situación de matrícula, becas, edad y trayectoria).
- El límite **global** de subida es de **10 MB**, incluido el laboratorio libre. No se requiere aumentar ese límite.
- `requirements.txt`, `reporting.py`, `app.py`, este README y `.streamlit/config.toml` se actualizaron de manera coordinada.
- Los informes PDF y Excel siguen disponibles en el módulo 6; solo incluyen resultados calculados en la sesión.
- Los logotipos proporcionados de UNAM y CUAED se conservan en `assets/`. Su uso institucional debe respetar la autorización aplicable.

## Fuentes académicas y acceso directo

- [Student Performance — UCI](https://archive.ics.uci.edu/dataset/320/student+performance)
- [Predict Students' Dropout and Academic Success — UCI](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- [OULAD — fuente original complementaria](https://analyse.kmi.open.ac.uk/open_dataset) (no se usa como fuente de los casos 12–14 en esta versión).

## Alcance y cautelas

Las prácticas 12–14 **no miden clics en entornos virtuales**: se redefinieron para analizar variables curriculares y de permanencia de UCI. La aplicación descarga las fuentes originales al ejecutarse, por lo que necesita conectividad desde Streamlit Cloud. Los modelos y comparaciones son exploratorios, no se replican automáticamente los resultados de los artículos y no deben usarse para decisiones individuales automatizadas.

## Despliegue

Subir **todos los archivos de este ZIP** a la raíz de `main` en GitHub, conservando `.streamlit/config.toml` y `assets/`. Si Streamlit Cloud mantiene una sesión antigua, usar `Manage app → Reboot app`. Confirmar `v0.6.1` en la barra lateral y en un PDF recién generado.

## Comprobaciones sugeridas

1. Abrir los casos 12, 13 y 14 y confirmar que se carga la tabla sin solicitar archivos.
2. Descargar el CSV de cada caso y revisar sus variables.
3. Ejecutar una regresión, clasificación y agrupamiento, según el objetivo seleccionado.
4. Generar PDF y XLSX desde el módulo 6.
5. Verificar que el laboratorio libre mantiene 10 MB de carga.
