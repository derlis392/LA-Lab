# LA-Lab v0.2
Laboratorio Virtual de Analítica del Aprendizaje.

**Autor académico:** Dr. Derlis Hernández Lara. Línea de investigación: Tecnologías Inteligentes para la Transformación Digital de los Ecosistemas Educativos.

## Publicación en Streamlit Community Cloud
1. Sustituye `app.py`, `requirements.txt` y agrega `.streamlit/config.toml` en la raíz del repositorio GitHub.
2. La aplicación conectada a la rama `main` se actualizará al detectar el commit.
3. Para ejecutar localmente: `pip install -r requirements.txt` y `streamlit run app.py`.

## Casos y procedencia
Los **14 casos integrados son datos sintéticos**, creados mediante generación reproducible para practicar técnicas y preguntas de Learning Analytics. **No son 14 bases reales descargadas de la literatura.** Se eligió esta estrategia para asegurar disponibilidad, ligereza y ausencia de datos personales en el despliegue público. Los repositorios abiertos verificados se enlazan en la aplicación para futuras incorporaciones con sus licencias y documentación.

Fuentes abiertas para ampliación:
- Open University Learning Analytics Dataset: https://doi.org/10.24432/C5KK69 (CC BY 4.0)
- Student Performance: https://doi.org/10.24432/C5TG7T (CC BY 4.0)
- Predict Students' Dropout and Academic Success: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success (ver condiciones de reutilización)

## Identidad institucional
La cabecera muestra el nombre de UNAM y CUAED. Para añadir **los emblemas oficiales**, colocar los archivos institucionales autorizados en:
- `assets/unam.png`
- `assets/cuaed.png`

No se incluyen emblemas reconstruidos o no verificados. Su incorporación debe respetar los lineamientos de identidad y evitar sugerir aval institucional distinto de la adscripción académica.

## Limitaciones
- No hay almacenamiento permanente ni cuentas; los resultados se descargan por sesión.
- No cargar datos personales identificables o sensibles.
- El servicio gratuito no garantiza 50 usuarios concurrentes; realizar pruebas de carga.
- La interpretación es estadística basada en reglas, no una IA generativa.
- No sustituye revisión metodológica, validación de indicadores o juicio pedagógico.
