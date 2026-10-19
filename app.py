"""LA-Lab v0.6.7 — Laboratorio abierto de Analítica del Aprendizaje."""
import io
import json
import zipfile
import urllib.request
from pathlib import Path
import reporting
from reporting import build_pdf, build_excel, explain_ml, describe_data, summarize_plot
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='LA-Lab | Analítica del Aprendizaje', page_icon='🎓', layout='wide')
ROOT = Path(__file__).parent
MAX_BYTES = 10 * 1024 * 1024
# Catorce prácticas basadas en cinco tablas originales de cuatro colecciones UCI.
SOURCES = {
 'student_mat': {'name':'Student Performance — Matemáticas', 'uci_id':320, 'url':'https://archive.ics.uci.edu/dataset/320/student+performance', 'article':'Cortez, P. & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance.', 'article_url':'https://repositorium.uminho.pt/entities/publication/42d177d6-868b-4e6c-8a89-e1e7fa41c1d0', 'license':'CC BY 4.0', 'target':'G3'},
 'student_por': {'name':'Student Performance — Portugués', 'uci_id':320, 'url':'https://archive.ics.uci.edu/dataset/320/student+performance', 'article':'Cortez, P. & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance.', 'article_url':'https://repositorium.uminho.pt/entities/publication/42d177d6-868b-4e6c-8a89-e1e7fa41c1d0', 'license':'CC BY 4.0', 'target':'G3'},
 'dropout': {'name':'Predict Students Dropout and Academic Success', 'uci_id':697, 'url':'https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success', 'article':'Realinho, V., Machado, J., Baptista, L. & Martins, M. (2022). Predicting Student Dropout and Academic Success. Data, 7(11), 146.', 'article_url':'https://doi.org/10.3390/data7110146', 'license':'CC BY 4.0', 'target':'Target'},
 'mathe': {'name':'MathE — Evaluación de matemáticas en educación superior', 'uci_id':1031, 'url':'https://archive.ics.uci.edu/dataset/1031/dataset+for+assessing+mathematics+learning+in+higher+educa', 'article':'Azevedo, B. F., Pacheco, M. F., Fernandes, F. P. & Pereira, A. I. (2024). Dataset of mathematics learning and assessment of higher education students using the MathE platform. Data in Brief, 53, 110236.', 'article_url':'https://doi.org/10.1016/j.dib.2024.110236', 'license':'CC BY 4.0', 'target':'Type of Answer'},
 'higher_ed': {'name':'Higher Education Students Performance Evaluation', 'uci_id':856, 'url':'https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation', 'article':'Yılmaz, N. & Şekeroğlu, B. (2020). Student Performance Classification Using Artificial Intelligence Techniques. Advances in Intelligent Systems and Computing, pp. 596–603.', 'article_url':'https://doi.org/10.1007/978-3-030-35249-3_76', 'license':'CC BY 4.0', 'target':'GRADE'},
 'oulad': {'name':'Open University Learning Analytics Dataset (OULAD)', 'url':'https://analyse.kmi.open.ac.uk/open_dataset', 'article':'Kuzilek, J., Hlosta, M. & Zdrahal, Z. (2017). Open University Learning Analytics dataset. Scientific Data, 4, 170171.', 'article_url':'https://doi.org/10.1038/sdata.2017.171', 'license':'CC BY 4.0', 'target':'final_result'},
}
CASES = [
 ('Rendimiento en matemáticas','student_mat','¿Cómo se relacionan las evaluaciones con la calificación final?','G3'),
 ('Ausencias y rendimiento','student_mat','¿Cómo se asocian las ausencias con G3?','G3'),
 ('Apoyo familiar y desempeño','student_mat','¿Qué diferencias descriptivas se observan según apoyos?','G3'),
 ('Tiempo de estudio','student_mat','¿Qué patrones aparecen entre studytime y G3?','G3'),
 ('Rendimiento en portugués','student_por','¿Cómo cambia el desempeño entre periodos?','G3'),
 ('Factores socioeducativos','student_por','¿Qué asociaciones aparecen entre características y rendimiento?','G3'),
 ('Comparación de modelos de regresión','student_por','¿Qué modelo predice G3 con menor error?','G3'),
 ('Abandono universitario','dropout','¿Qué variables se asocian con la clase de trayectoria?','Target'),
 ('Éxito académico','dropout','¿Qué patrones diferencian las categorías de resultado?','Target'),
 ('Clasificación multiclase','dropout','¿Cómo se comportan los clasificadores ante clases desbalanceadas?','Target'),
 ('Segmentación de perfiles','dropout','¿Qué grupos exploratorios se obtienen sin utilizar la etiqueta?','Target'),
 ('Respuestas y dificultades en matemáticas','mathe','¿Cómo varía la proporción de respuestas correctas e incorrectas según el tema y el nivel de las preguntas?','Type of Answer'),
 ('Hábitos de estudio y desempeño universitario','higher_ed','¿Qué asociaciones se observan entre hábitos de estudio, condiciones académicas y calificación final?','GRADE'),
 ('Factores de permanencia universitaria','dropout','¿Cómo se relacionan las condiciones de matrícula y los apoyos económicos con la trayectoria académica?','Target'),
]

def _read_mathe_archive(payload):
    """Read the actual MathE table from the original UCI archive, not an invented dataset."""
    if len(payload) > 45 * 1024 * 1024:
        raise ValueError('El archivo de origen excede el límite de seguridad de 45 MB comprimidos.')
    candidates = []
    def visit(data, depth=0):
        if depth > 2:
            return
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                name = member.filename
                if member.is_dir() or name.startswith('__MACOSX/') or '/.' in name:
                    continue
                if member.file_size > 90 * 1024 * 1024:
                    continue
                suffix = Path(name).suffix.lower()
                if suffix not in ('.csv', '.xlsx', '.xls', '.zip'):
                    continue
                raw = archive.read(member)
                if suffix == '.zip':
                    try:
                        visit(raw, depth + 1)
                    except (ValueError, zipfile.BadZipFile):
                        continue
                else:
                    try:
                        if suffix in ('.xlsx', '.xls'):
                            frame = pd.read_excel(io.BytesIO(raw))
                        else:
                            frame = None
                            for sep in (',', ';', '\t'):
                                for encoding in ('utf-8-sig', 'latin-1'):
                                    try:
                                        trial = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding, low_memory=False)
                                        if trial.shape[1] >= 3:
                                            frame = trial
                                            break
                                    except (UnicodeError, pd.errors.ParserError):
                                        continue
                                if frame is not None:
                                    break
                        if frame is not None and len(frame) >= 50 and frame.shape[1] >= 3:
                            score = (int(any('answer' in str(c).lower() for c in frame.columns)), len(frame))
                            candidates.append((score, frame))
                    except (ValueError, ImportError, OSError, pd.errors.ParserError):
                        continue
    visit(payload)
    if not candidates:
        raise ValueError('El ZIP descargado no contiene una tabla MathE reconocible (CSV o Excel).')
    return max(candidates, key=lambda item: item[0])[1].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=86400)
def load_original(source_id):
    # Descarga bajo demanda en el servidor Streamlit; nunca se sustituyen datos reales por sintéticos.
    from ucimlrepo import fetch_ucirepo
    source = SOURCES[source_id]
    if source_id.startswith('student_'):
        url='https://archive.ics.uci.edu/static/public/320/student+performance.zip'
        with urllib.request.urlopen(url,timeout=25) as response:
            payload=response.read()
        if len(payload)>3_000_000: raise ValueError('Descarga mayor de lo esperado.')
        def extract_csv(data, filename, depth=0):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for item in z.namelist():
                    if item.lower().endswith(filename):
                        return pd.read_csv(io.BytesIO(z.read(item)),sep=';')
                if depth<2:
                    for item in z.namelist():
                        if item.lower().endswith('.zip'):
                            try: return extract_csv(z.read(item),filename,depth+1)
                            except (KeyError,ValueError,zipfile.BadZipFile): pass
            raise KeyError(f'No se encontró {filename} en el ZIP original.')
        return extract_csv(payload,'student-mat.csv' if source_id=='student_mat' else 'student-por.csv')
    if source_id == 'mathe':
        # Copia fiel del CSV original MathE, provisto desde el archivo UCI por el usuario.
        # No depende de URL externas ni del límite de carga del laboratorio libre.
        csv_path = ROOT / 'data' / 'mathe_original.csv'
        if not csv_path.is_file():
            raise FileNotFoundError('Falta data/mathe_original.csv en el repositorio. Sube el paquete completo.')
        frame = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        expected = {'Student ID', 'Student Country', 'Question ID', 'Type of Answer', 'Question Level', 'Topic', 'Subtopic', 'Keywords'}
        if not expected.issubset(frame.columns) or len(frame) != 9546:
            raise ValueError('El archivo MathE local no coincide con el conjunto original verificado (9546 registros, 8 variables).')
        return frame
    ds = fetch_ucirepo(id=source['uci_id'])
    frames = [part.reset_index(drop=True) for part in (ds.data.features, ds.data.targets) if isinstance(part, pd.DataFrame) and not part.empty]
    if not frames: raise ValueError('La fuente no devolvió una tabla de datos.')
    df = pd.concat(frames,axis=1)
    return df.loc[:,~df.columns.duplicated()].copy()


def parse_upload(upload):
    if upload.size > MAX_BYTES:
        raise ValueError('El archivo supera el límite de 10 MB.')
    name = upload.name.lower()
    raw = upload.getvalue()
    if name.endswith(('.xlsx','.xls')):
        return pd.read_excel(io.BytesIO(raw))
    if name.endswith('.json'):
        obj = json.loads(raw.decode('utf-8-sig'))
        if not isinstance(obj,list) or not all(isinstance(x,dict) for x in obj):
            raise ValueError('JSON debe contener una lista de objetos (registros).')
        return pd.DataFrame(obj)
    if name.endswith('.parquet'):
        return pd.read_parquet(io.BytesIO(raw))
    if name.endswith(('.csv','.tsv','.txt')):
        sep = '\t' if name.endswith('.tsv') else None
        for enc in ['utf-8-sig','latin-1']:
            try:
                return pd.read_csv(io.BytesIO(raw),sep=sep if sep else None,engine='python',encoding=enc)
            except UnicodeError:
                continue
        raise ValueError('No fue posible identificar la codificación.')
    raise ValueError('Formato no compatible.')

def csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8-sig')

def interpret_dataset(df):
    n,m = df.shape
    nums = df.select_dtypes(include='number').columns.tolist()
    cats = [c for c in df.columns if c not in nums]
    missing = int(df.isna().sum().sum())
    dup = int(df.duplicated().sum())
    lines = [f'La base contiene **{n:,} registros y {m} variables**: {len(nums)} numéricas y {len(cats)} no numéricas.',
             f'Se identificaron **{missing:,} celdas faltantes** y **{dup:,} filas completamente duplicadas**.']
    if nums:
        sample = nums[:4]
        lines.append('Variables numéricas detectadas: '+', '.join(f'`{c}`' for c in sample)+'. Se pueden estudiar distribuciones, dispersión y asociaciones, según su significado.')
    if cats:
        lines.append('Variables no numéricas: '+', '.join(f'`{c}`' for c in cats[:4])+'. Permiten comparar frecuencias y grupos cuando representan categorías.')
    if n < 30: lines.append('**Precaución:** el tamaño de la muestra es pequeño; las estimaciones pueden ser inestables.')
    if missing: lines.append('Antes de analizar, revise el patrón de datos faltantes y documente cualquier tratamiento.')
    lines.append('**Qué se puede esperar:** descripciones, comparaciones y visualizaciones exploratorias. La estructura de datos por sí sola no permite inferir causalidad ni medir directamente el aprendizaje.')
    return '\n\n'.join(lines)

def summarize_numeric(s):
    s = pd.to_numeric(s,errors='coerce').dropna()
    if s.empty: return 'No hay observaciones numéricas válidas.'
    return f'En **{len(s)} observaciones válidas**, el promedio es **{s.mean():.2f}**, la mediana **{s.median():.2f}** y los valores van de **{s.min():.2f}** a **{s.max():.2f}**. La desviación estándar es **{s.std():.2f}**.' if len(s)>1 else f'Hay un solo valor válido: {s.iloc[0]:.2f}.'

def text_box(text):
    st.markdown('##### Interpretación de los hallazgos')
    st.info(text)

st.markdown('''<style>
.block-container {padding-top:1.7rem;max-width:1300px}
h1,h2,h3 {letter-spacing:-.02em}
[data-testid="stSidebar"] {background:#f1f5fa}
</style>''',unsafe_allow_html=True)

with st.sidebar:
    st.title('🎓 LA-Lab')
    st.caption('Laboratorio Virtual de Analítica del Aprendizaje · v0.6.7')
    mode = st.radio('Modalidad', ['Casos guiados','Datos propios'],key='mode')
    idx = None
    if mode == 'Casos guiados':
        idx = st.selectbox('Caso de estudio',range(len(CASES)),format_func=lambda i:f'{i+1:02d}. {CASES[i][0]}')
    else:
        upload = st.file_uploader('Subir datos (máximo 10 MB)',type=['csv','tsv','txt','xlsx','xls','json','parquet'])
    page = st.radio('Módulo',['Inicio','1. Datos','2. Limpieza','3. Indicadores','4. Visualización','5. Interpretación','6. Informe','7. Minería de datos'])
    st.warning('Utiliza datos sintéticos o autorizados y desidentificados. No cargues datos personales sensibles.')

header_left, header_right = st.columns([3.5, 1.6], vertical_alignment="top")
with header_left:
    st.title('🎓 LA-Lab')
    st.caption('Tecnologías Inteligentes para la Transformación Digital de los Ecosistemas Educativos')
    st.caption('Dr. Derlis Hernández Lara · CUAED · Universidad Nacional Autónoma de México')
    research_logo = ROOT/'assets'/'linea_investigacion.svg'
    if research_logo.exists():
        with st.expander('Ver figura de la línea de investigación'):
            st.image(str(research_logo), caption='Tecnologías Inteligentes para la Transformación Digital de los Ecosistemas Educativos', width=420)
with header_right:
    st.markdown("<div style='height:58px'></div>", unsafe_allow_html=True)
    logos = ROOT/'assets'
    unam, cuaed = logos/'unam.png', logos/'cuaed.png'
    if unam.exists() and cuaed.exists():
        # Mostrar imágenes originales a escala reducida sin reescalar archivos fuente.
        # Para zoom sin pixelación se requieren SVG institucionales oficiales.
        col_unam, col_cuaed = st.columns([1, 2.2], vertical_alignment="center")
        with col_unam: st.image(str(unam), width=78)
        with col_cuaed: st.image(str(cuaed), width=156)
    else:
        st.caption('Identidad institucional: emblemas pendientes')

if mode=='Casos guiados':
    title, source_id, question, suggested_target = CASES[idx]
    source = SOURCES[source_id]
    description = source['name']
    try:
        df_original = load_original(source_id)
        if idx >= 11:
            st.success('Práctica guiada cargada automáticamente con datos originales de UCI. No requiere subir archivos.')
            st.caption('Los casos 12, 13 y 14 usan tres conjuntos originales diferentes: MathE (UCI 1031), Higher Education (UCI 856) y abandono universitario (UCI 697).')
            st.download_button('Descargar datos originales de esta práctica (.csv)', csv_bytes(df_original), f'la_lab_caso_{idx+1:02d}.csv', 'text/csv')
    except Exception as ex:
        df_original = None
        st.error('No fue posible obtener la fuente original: '+str(ex))
        if source_id == 'mathe':
            st.info('El caso 12 utiliza el archivo original incluido en data/mathe_original.csv. Verifica que esa carpeta se haya subido a GitHub.')
    if idx == 11:
        st.caption('Variables recomendadas: Topic, Subtopic, Question Level y Type of Answer. Cada fila es una respuesta, no un estudiante independiente; evita Student ID y Question ID como predictores automáticos.')
    elif idx == 12:
        st.caption('Explora las variables de hábitos de estudio y la calificación GRADE; el conjunto tiene 145 observaciones. Las columnas codificadas requieren consultar el diccionario de UCI.')
    elif idx == 13:
        st.caption('Variables recomendadas: Tuition fees up to date, Scholarship holder, Age at enrollment y Target. Analiza asociaciones sin inferir causalidad; evita incluir datos posteriores a la matrícula en modelos de alerta temprana.')
    st.caption('FUENTE ORIGINAL: '+source['name']+' · '+source['license'])
    with st.expander('Ficha científica, artículo y descarga del conjunto original',expanded=True):
        st.write('**Pregunta:** '+question)
        st.write('**Fuente:** '+source['name'])
        st.link_button('Abrir fuente y descargar datos originales',source['url'])
        st.write('**Publicación científica de referencia:** '+source['article'])
        if source.get('article_url'):
            st.link_button('Consultar publicación científica',source['article_url'])
        else:
            st.caption('Referencia bibliográfica disponible; enlace directo al artículo pendiente de verificación.')
        st.caption('El DOI de la ficha UCI identifica el conjunto de datos; el enlace de publicación científica identifica el trabajo académico. Las prácticas son análisis propios, no reproducciones validadas de esos trabajos.')
else:
    if upload is None:
        df_original = None
    else:
        try:
            df_original = parse_upload(upload)
            if df_original.empty or len(df_original.columns)==0: raise ValueError('El archivo no contiene una tabla con registros y columnas.')
            if len(df_original)>100000: raise ValueError('Máximo inicial: 100 000 filas. Reduce el archivo para usar el laboratorio público.')
            if len(df_original.columns)>100: raise ValueError('Máximo inicial: 100 columnas.')
        except Exception as ex:
            st.error(f'No se pudo cargar el archivo: {ex}')
            df_original = None
    title, description, question = 'Datos propios','Archivo proporcionado por el usuario','¿Qué información puede obtenerse de estas variables?'
    suggested_target = None

source_key = f'{mode}-{idx if idx is not None else ""}-{upload.name if mode=="Datos propios" and upload else ""}-{upload.size if mode=="Datos propios" and upload else ""}'
if st.session_state.get('source_key')!=source_key:
    st.session_state.source_key=source_key
    st.session_state.cleaned=None
    st.session_state.last_cleaning='No se han aplicado transformaciones.'
    st.session_state.ml_summary='Sin modelo entrenado.'
    st.session_state.plot_summary='Sin gráfico analizado.'
    st.session_state.ml_payload=None
    st.session_state.plot_payload=None
df = st.session_state.cleaned if st.session_state.cleaned is not None else df_original

if page=='Inicio':
    st.header('Bienvenido al laboratorio')
    st.write('Selecciona un caso guiado o carga una base propia. Recorre los módulos para preparar, analizar, visualizar e interpretar los datos.')
    a,b=st.columns(2)
    with a:
        st.subheader('📚 Laboratorio guiado')
        st.write('14 prácticas sobre cinco tablas originales de cuatro colecciones UCI, con preguntas de investigación y actividades orientadas a la analítica educativa.')
    with b:
        st.subheader('📂 Laboratorio libre')
        st.write('Carga CSV, TSV, TXT delimitado, Excel, JSON de registros o Parquet y analiza variables mediante controles gráficos.')
    st.markdown('##### Procedencia y alcance')
    st.info('Los 14 casos son **prácticas sobre dos colecciones UCI de datos reales** (Student Performance y Predict Students Dropout). OULAD se mantiene como fuente complementaria para investigaciones que requieran actividad virtual. Los artículos citados no constituyen una validación automática de los resultados obtenidos aquí.')
    st.markdown('##### Fuentes abiertas para ampliar el laboratorio')
    st.markdown('- [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset) — datos reales de cursos y actividad virtual, CC BY 4.0.\n- [Student Performance (UCI)](https://doi.org/10.24432/C5TG7T) — desempeño académico de secundaria, CC BY 4.0.\n- [Predict Students’ Dropout and Academic Success (UCI)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) — trayectoria y resultados de educación superior; consultar condiciones antes de reutilizar.')
    st.caption('Las 14 prácticas UCI cargan datos reales desde la fuente original cuando el servidor tiene acceso; no es necesario cargar un ZIP de OULAD. No se generan datos ficticios como sustitutos.')
elif df is None:
    st.info('Carga primero un archivo desde el panel izquierdo para comenzar.')
elif page=='1. Datos':
    st.header('1. Exploración y comprensión de los datos')
    st.write(f'**Caso:** {title}. {description}')
    a,b,c,d=st.columns(4)
    a.metric('Registros',len(df)); b.metric('Variables',len(df.columns)); c.metric('Celdas vacías',int(df.isna().sum().sum())); d.metric('Duplicados',int(df.duplicated().sum()))
    st.dataframe(df.head(100),use_container_width=True)
    schema=pd.DataFrame({'Variable':df.columns,'Tipo técnico':df.dtypes.astype(str).values,'Faltantes':df.isna().sum().values,'Valores únicos':[df[c].nunique(dropna=True) for c in df.columns]})
    st.dataframe(schema,use_container_width=True,hide_index=True)
    text_box(interpret_dataset(df))
    st.caption('Los tipos técnicos no sustituyen la definición conceptual de las variables ni sus unidades.')
elif page=='2. Limpieza':
    st.header('2. Preparación y limpieza')
    st.write('Inspecciona los problemas y aplica transformaciones documentadas. El conjunto original se conserva para poder restablecerlo.')
    st.dataframe(pd.DataFrame({'Variable':df.columns,'Faltantes':df.isna().sum().values}),hide_index=True,use_container_width=True)
    st.write(f'Filas duplicadas completas: **{df.duplicated().sum()}**')
    remove=st.checkbox('Eliminar filas duplicadas',value=True)
    missing=st.selectbox('Tratamiento de faltantes',['Conservar','Eliminar filas con faltantes','Imputar numéricas con mediana y categóricas con moda'])
    if st.button('Aplicar transformaciones',type='primary'):
        cleaned=df.copy()
        before=len(cleaned)
        if remove: cleaned=cleaned.drop_duplicates()
        if missing=='Eliminar filas con faltantes': cleaned=cleaned.dropna()
        elif missing.startswith('Imputar'):
            for c in cleaned.columns:
                if cleaned[c].isna().any():
                    valid=cleaned[c].dropna()
                    if valid.empty: continue
                    value=valid.median() if pd.api.types.is_numeric_dtype(cleaned[c]) else valid.mode().iloc[0]
                    cleaned[c]=cleaned[c].fillna(value)
        st.session_state.cleaned=cleaned
        st.session_state.ml_payload=None
        st.session_state.plot_payload=None
        st.session_state.last_cleaning=f'Filas antes: {before}; después: {len(cleaned)}. Duplicados: {remove}. Faltantes: {missing}.'
        st.success(st.session_state.get('last_cleaning','Sin operaciones registradas'))
        st.rerun()
    if st.button('Restablecer datos originales'):
        st.session_state.cleaned=None
        st.session_state.ml_payload=None
        st.session_state.plot_payload=None
        st.session_state.last_cleaning='Datos originales restablecidos.'
        st.rerun()
    st.download_button('Descargar tabla actual (CSV)',csv_bytes(df),'la_lab_datos_limpios.csv','text/csv')
    text_box(f'El conjunto actual tiene **{len(df)} filas**, **{int(df.isna().sum().sum())} valores faltantes** y **{int(df.duplicated().sum())} duplicados completos**. {st.session_state.get("last_cleaning","Sin operaciones registradas")} Antes de imputar, revisa si los datos faltan sistemáticamente; la imputación puede alterar la distribución.')
elif page=='3. Indicadores':
    st.header('3. Estadística e indicadores')
    numeric=df.select_dtypes(include='number').columns.tolist()
    if not numeric: st.warning('No hay variables numéricas para analizar.'); st.stop()
    var=st.selectbox('Variable numérica',numeric)
    s=pd.to_numeric(df[var],errors='coerce').dropna()
    a,b,c,d=st.columns(4)
    a.metric('Media',f'{s.mean():.2f}' if len(s) else '—')
    b.metric('Mediana',f'{s.median():.2f}' if len(s) else '—')
    c.metric('Desv. estándar',f'{s.std():.2f}' if len(s)>1 else '—')
    d.metric('Datos válidos',len(s))
    st.dataframe(df[numeric].describe().T,use_container_width=True)
    if 'entregas' in df and pd.api.types.is_numeric_dtype(df['entregas']):
        st.metric('Promedio de entregas registradas',f'{df["entregas"].mean():.2f}')
        st.caption('Cantidad promedio, no porcentaje de cumplimiento salvo que se conozca el total esperado.')
    st.caption('Media: tendencia central; mediana: valor intermedio; desviación estándar: variabilidad. Revisa si la variable representa cantidad, categoría codificada o escala ordinal.')
    text_box(summarize_numeric(df[var])+' Estas cifras describen la muestra; su relevancia educativa depende de la escala, el periodo y el contexto. No confundir correlación con causalidad ni frecuencia de actividad con aprendizaje.')
elif page=='4. Visualización':
    st.header('4. Visualización exploratoria')
    numeric=df.select_dtypes(include='number').columns.tolist()
    cats=df.select_dtypes(exclude='number').columns.tolist()
    kinds=['Histograma','Diagrama de dispersión','Diagrama de caja','Frecuencia de categorías','Matriz de correlación']
    kind=st.selectbox('Tipo de gráfico',kinds)
    if kind=='Histograma' and numeric:
        x=st.selectbox('Variable',numeric); fig=px.histogram(df,x=x,nbins=25)
        st.plotly_chart(fig,use_container_width=True)
        st.session_state.plot_summary=f'Distribución de {x}'
        st.session_state.plot_payload={'tipo':'histograma','x':x}
        text_box(f'El histograma muestra la distribución de **{x}**. {summarize_numeric(df[x])} Examina concentración, asimetría y posibles valores extremos.')
    elif kind=='Diagrama de dispersión' and len(numeric)>=2:
        x=st.selectbox('Eje X',numeric); y=st.selectbox('Eje Y',[c for c in numeric if c!=x]); fig=px.scatter(df,x=x,y=y,opacity=.7)
        st.plotly_chart(fig,use_container_width=True)
        pair=df[[x,y]].dropna()
        r=pair[x].corr(pair[y]) if len(pair)>2 and pair[x].nunique()>1 and pair[y].nunique()>1 else np.nan
        msg=f'La correlación de Pearson entre **{x}** y **{y}** es **{r:.2f}**.' if pd.notna(r) else 'No es posible calcular una correlación válida.'
        st.session_state.plot_summary=msg
        st.session_state.plot_payload={'tipo':'dispersion','x':x,'y':y}
        text_box(msg+' La asociación no demuestra causalidad; evalúa valores extremos, linealidad y posibles factores de confusión.')
    elif kind=='Diagrama de caja' and numeric:
        y=st.selectbox('Variable',numeric); group=st.selectbox('Agrupar por (opcional)',['Sin grupos']+cats)
        fig=px.box(df,y=y,x=group if group!='Sin grupos' else None)
        st.plotly_chart(fig,use_container_width=True)
        st.session_state.plot_summary=f'Diagrama de caja de {y} por {group}'
        st.session_state.plot_payload={'tipo':'caja','y':y,'grupo':group}
        text_box(f'El diagrama de caja permite comparar medianas, dispersión y observaciones extremas de **{y}**'+(f' por **{group}**.' if group!='Sin grupos' else '.')+' Las diferencias descriptivas no implican efectos causales.')
    elif kind=='Frecuencia de categorías' and cats:
        x=st.selectbox('Variable categórica',cats)
        counts=df[x].astype('string').fillna('(Faltante)').value_counts().head(20).rename_axis(x).reset_index(name='frecuencia')
        st.plotly_chart(px.bar(counts,x=x,y='frecuencia'),use_container_width=True)
        st.session_state.plot_summary=f'Frecuencias de {x}'
        st.session_state.plot_payload={'tipo':'categorias','x':x}
        text_box(f'La categoría más frecuente de **{x}** es **{counts.iloc[0,0]}**, con **{counts.iloc[0,1]} registros**. Se muestran como máximo 20 categorías.')
    elif kind=='Matriz de correlación' and len(numeric)>=2:
        corr=df[numeric].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale='RdBu_r'),use_container_width=True)
        st.session_state.plot_summary='Matriz de correlación calculada para '+str(len(numeric))+' variables numéricas.'
        st.session_state.plot_payload={'tipo':'correlacion','columnas':numeric[:15]}
        text_box('La matriz muestra asociaciones lineales entre variables numéricas. Correlaciones elevadas pueden reflejar variables redundantes, relaciones mecánicas o patrones que requieren investigación adicional.')
    else:
        st.warning('El archivo no tiene suficientes variables compatibles con este tipo de gráfico.')
elif page=='5. Interpretación':
    st.header('5. Interpretación y decisiones educativas')
    st.write('**Pregunta orientadora:** '+question)
    text_box(describe_data(df,question,suggested_target)+'\n\n'+summarize_plot(df,st.session_state.get('plot_payload'))+'\n\n'+explain_ml(st.session_state.get('ml_payload')))
    st.write('**Actividad de reflexión**')
    choice=st.radio('¿Qué conclusión es metodológicamente adecuada?', ['Selecciona una respuesta','Las correlaciones demuestran que una variable causa otra.','Los patrones son descriptivos y requieren contraste contextual.','Una variable de accesos mide directamente el aprendizaje.'])
    if st.button('Verificar respuesta'):
        if choice.startswith('Los patrones'): st.success('Correcto. Una interpretación responsable distingue descripción, asociación y causalidad.')
        elif choice=='Selecciona una respuesta': st.warning('Selecciona una respuesta primero.')
        else: st.error('Revisa la diferencia entre actividad registrada, aprendizaje y causalidad.')
    st.write('**Propuesta de acción:** formula una decisión pedagógica que podría investigarse, no una intervención automática sobre personas.')
    st.text_area('Redacta tu interpretación y recomendación',key='reflection',height=120)
    if st.session_state.get('reflection'):
        st.download_button('Descargar reflexión (.txt)',st.session_state.reflection,'reflexion_la_lab.txt','text/plain')
elif page=='6. Informe':
    st.header('6. Informe académico integral')
    st.write('Exporta resultados, figuras, interpretaciones, limitaciones, recomendaciones y referencias. Solo se incluyen análisis efectivamente ejecutados en esta sesión.')
    ml=st.session_state.get('ml_payload')
    plot=st.session_state.get('plot_payload')
    overview=describe_data(df,question, suggested_target)
    st.info(overview)
    if ml:
        st.subheader('Interpretación del experimento más reciente')
        st.info(explain_ml(ml))
        st.write('**Recomendación metodológica:** compara con el modelo de referencia, valida la estabilidad y evita decisiones automatizadas sobre estudiantes.')
    else:
        st.warning('Todavía no se ha ejecutado un modelo de minería de datos en esta sesión; el informe no inventará resultados.')
    if plot:
        st.caption('Visualización exploratoria registrada: '+st.session_state.get('plot_summary',''))
    else:
        st.caption('No hay visualización exploratoria registrada. Se incorporarán gráficos descriptivos calculados a partir de la base.')
    metadata={'caso':title,'fuente':description if mode=='Casos guiados' else 'Archivo proporcionado por el usuario',
              'pregunta':question,'limpieza':st.session_state.get('last_cleaning','Sin transformaciones'),
              'visualizacion':st.session_state.get('plot_summary','No ejecutada'),
              'fuente_url':source['url'] if mode=='Casos guiados' else '',
              'articulo':source['article'] if mode=='Casos guiados' else '',
              'articulo_url':source['article_url'] if mode=='Casos guiados' else '',
              'licencia':source.get('license','') if mode=='Casos guiados' else 'Verificar autorización del archivo',
              'reflexion':st.session_state.get('reflection','')}
    st.download_button('Descargar datos actuales (.csv)',csv_bytes(df),'datos_la_lab.csv','text/csv')
    txt='LA-Lab v0.6.7 | '+title+'\n'+overview+'\n\n'+(explain_ml(ml) if ml else 'Sin experimento de ML')+'\n'+str(metadata)
    st.download_button('Descargar síntesis (.txt)',txt.encode('utf-8'),'informe_la_lab_v061.txt','text/plain')
    st.caption(f'Generador de informes cargado: {getattr(reporting, "REPORT_VERSION", "ANTIGUO")} · archivo: {Path(reporting.__file__).name}')
    try:
        if getattr(reporting, 'REPORT_VERSION', None) != '0.6':
            st.error('VERSIONES INCOMPATIBLES: actualiza reporting.py junto con app.py en la raíz del repositorio y reinicia Streamlit. No se descargará un PDF antiguo.')
        else:
            pdf=build_pdf(df,metadata,ml,plot)
            st.download_button('Descargar informe académico v0.6.7 (.pdf)',pdf,'informe_la_lab_v067.pdf','application/pdf')
    except Exception as exc:
        st.error(f'No se pudo generar el PDF: {exc}')
    try:
        xlsx=build_excel(df,metadata,ml)
        st.download_button('Descargar libro de análisis (.xlsx)',xlsx,'analisis_la_lab_v061.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as exc:
        st.error(f'No se pudo generar el Excel: {exc}')
    text_box('El informe incorpora análisis registrados, visualizaciones reproducibles, interpretación y sugerencias condicionadas. Las recomendaciones requieren revisión humana y no constituyen diagnóstico individual.')

elif page=='7. Minería de datos':
    st.header('7. Minería de datos y aprendizaje automático')
    st.caption('Experimentos exploratorios. No usar para decisiones automatizadas sobre estudiantes.')
    from sklearn.model_selection import train_test_split, cross_validate, KFold, StratifiedKFold
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
    from sklearn.dummy import DummyRegressor, DummyClassifier
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix,
                                 classification_report, mean_absolute_error, mean_squared_error,
                                 r2_score, silhouette_score)
    import plotly.graph_objects as go
    st.markdown('**Guía rápida:** clasificación predice categorías; regresión predice cantidades; agrupamiento busca similitudes sin etiquetas.')
    kind=st.selectbox('Tipo de tarea',['Clasificación','Regresión','Agrupamiento'])
    options=list(df.columns)
    numeric_options=df.select_dtypes(include='number').columns.tolist()
    if kind!='Agrupamiento':
        target=st.selectbox('Variable objetivo',options,index=options.index(suggested_target) if suggested_target in options else len(options)-1)
        candidates=[c for c in options if c!=target]
        recommended=[c for c in ['studytime','failures','absences','Medu','Fedu'] if c in candidates]
        defaults=recommended if recommended else candidates[:min(6,len(candidates))]
        predictors=st.multiselect('Variables predictoras',candidates,default=defaults,max_selections=12)
    else:
        target=None
        recommended=[c for c in ['studytime','failures','absences','Medu','Fedu'] if c in options]
        predictors=st.multiselect('Variables para agrupar',options,default=recommended if recommended else numeric_options[:5],max_selections=12)
    choices={'Clasificación':['Regresión logística','Árbol de decisión','Random Forest','SVM','KNN','Gradient Boosting'],
             'Regresión':['Regresión lineal','Ridge','Árbol de regresión','Random Forest'],
             'Agrupamiento':['K-Means','Jerárquico','DBSCAN']}
    algorithm=st.selectbox('Algoritmo',choices[kind])
    explanations={'Regresión logística':'Clasificador lineal probabilístico; puede utilizar ponderación de clases.',
                  'Random Forest':'Conjunto de árboles; capta relaciones no lineales, pero requiere comprobar generalización.',
                  'K-Means':'Agrupa por distancia a centroides. Requiere escalamiento y seleccionar k.',
                  'Jerárquico':'Construye grupos por proximidad; sensible a la distancia y al escalamiento.',
                  'DBSCAN':'Identifica regiones densas y posibles observaciones de ruido (-1).',
                  'Regresión lineal':'Modelo lineal interpretable; examina supuestos y residuos.'}
    st.caption(explanations.get(algorithm,'Compara este algoritmo con un modelo de referencia y revisa las métricas por grupo.'))
    seed=st.number_input('Semilla aleatoria',0,100000,42)
    if kind!='Agrupamiento':
        test_pct=st.slider('Porcentaje de prueba',15,40,20,5)
        compare=st.checkbox('Comparar todos los algoritmos de esta tarea (misma partición)',value=False)
        cv=st.checkbox('Validación cruzada en entrenamiento (3 particiones)',value=False)
        balanced=st.checkbox('Ponderar clases cuando sea posible',value=False) if kind=='Clasificación' else False
    else:
        clusters=st.slider('Número de grupos (si aplica)',2,8,3)
        compare=st.checkbox('Comparar silhouette para varios valores de k (K-Means)',value=False)
    st.caption('Preprocesamiento dentro del pipeline en tareas supervisadas. Límite de 5 000 registros y 12 predictores para proteger el servidor.')
    if st.button('Ejecutar análisis',type='primary'):
        try:
            if not predictors: raise ValueError('Selecciona al menos una variable.')
            if target=='G3' and any(c in predictors for c in ['G1','G2']):
                st.warning('Atención a la temporalidad: G1 y G2 son calificaciones previas. Pueden ser predictores válidos al final del curso, pero no están disponibles al inicio; define el momento de predicción para evitar fuga temporal.')
            work=df[predictors+([target] if target else [])].copy()
            if target: work=work.dropna(subset=[target])
            if kind=='Clasificación':
                freq=work[target].value_counts()
                if len(freq)>1 and freq.min()<2:
                    raise ValueError(f'La variable {target} tiene categorías con una sola observación; no admite partición estratificada. Si es una calificación numérica, considera regresión o crea categorías justificadas antes de cargar los datos.')
            if len(work)>5000: work=work.sample(5000,random_state=int(seed))
            if len(work)<20: raise ValueError('Se necesitan al menos 20 observaciones válidas.')
            X=work[predictors].copy()
            for c in X.columns:
                if pd.api.types.is_datetime64_any_dtype(X[c]): X[c]=X[c].astype(str)
            nums=X.select_dtypes(include='number').columns.tolist()
            cats=[c for c in predictors if c not in nums]
            transforms=[]
            if nums: transforms.append(('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums))
            if cats: transforms.append(('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats))
            pre=ColumnTransformer(transforms)
            comparison_rows=None
            cv_details=None
            payload={'tarea':kind,'algoritmo':algorithm,'variables':list(predictors),'objetivo':target,'semilla':int(seed),'n':len(work),'porcentaje_prueba':test_pct if kind!='Agrupamiento' else None}
            if kind=='Clasificación':
                y=work[target].astype(str)
                if not 2<=y.nunique()<=30: raise ValueError('La clasificación requiere entre 2 y 30 categorías. Una calificación numérica con muchos valores corresponde a regresión.')
                models={'Regresión logística':LogisticRegression(max_iter=1000,class_weight='balanced' if balanced else None),
                        'Árbol de decisión':DecisionTreeClassifier(max_depth=6,class_weight='balanced' if balanced else None,random_state=int(seed)),
                        'Random Forest':RandomForestClassifier(n_estimators=80,max_depth=10,class_weight='balanced' if balanced else None,random_state=int(seed),n_jobs=1),
                        'SVM':SVC(class_weight='balanced' if balanced else None),
                        'KNN':KNeighborsClassifier(),
                        'Gradient Boosting':GradientBoostingClassifier(random_state=int(seed))}
                counts_all=y.value_counts()
                if counts_all.min()<2: raise ValueError('Cada clase necesita al menos dos observaciones para particionar.')
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_pct/100,random_state=int(seed),stratify=y)
                payload.update({'n_entrenamiento':len(Xtr),'n_prueba':len(Xte),'preprocesamiento_interno':'Imputación de numéricas por mediana, escalado estándar; imputación de categóricas por moda y codificación one-hot (según tipos presentes). Ajuste dentro del pipeline de entrenamiento.'})
                pipe=Pipeline([('prep',pre),('model',models[algorithm])]);pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
                labels=sorted(y.unique())
                cm=confusion_matrix(yte,pred,labels=labels)
                acc=accuracy_score(yte,pred); bacc=balanced_accuracy_score(yte,pred); f1=f1_score(yte,pred,average='macro',zero_division=0)
                baseline=DummyClassifier(strategy='most_frequent').fit(Xtr,ytr)
                baseline_acc=accuracy_score(yte,baseline.predict(Xte))
                result=f'Clasificación {algorithm}: n prueba={len(yte)}, accuracy={acc:.3f}, balanced accuracy={bacc:.3f}, F1 macro={f1:.3f}; baseline accuracy={baseline_acc:.3f}.'
                if yte.value_counts(normalize=True).max()>.75: st.warning('Clases desbalanceadas: consulta recall por clase y balanced accuracy; accuracy puede ser engañosa.')
                st.subheader('Matriz de confusión')
                fig=px.imshow(cm,x=labels,y=labels,text_auto=True,labels={'x':'Predicción','y':'Clase real','color':'Casos'},color_continuous_scale='Blues',aspect='auto')
                st.plotly_chart(fig,use_container_width=True)
                report=pd.DataFrame(classification_report(yte,pred,labels=labels,output_dict=True,zero_division=0)).T
                payload.update({'reales':yte.astype(str).tolist(),'predicciones':pd.Series(pred).astype(str).tolist(),'etiquetas':list(map(str,labels)),'matriz':cm.tolist(),'metricas':{'Accuracy':acc,'Balanced accuracy':bacc,'F1 macro':f1,'Baseline accuracy':baseline_acc},'por_clase':report.loc[labels,['precision','recall','f1-score','support']].reset_index().rename(columns={'index':'Clase'}).to_dict('records')})
                st.subheader('Desempeño por clase')
                st.dataframe(report.loc[labels,['precision','recall','f1-score','support']].round(3),use_container_width=True)
                st.plotly_chart(px.bar(report.loc[labels].reset_index(names='Clase'),x='Clase',y=['precision','recall','f1-score'],barmode='group',title='Métricas por clase'),use_container_width=True)
                st.caption(f'Baseline mayoritario: accuracy={baseline_acc:.3f}. Un resultado superior al baseline no implica validez para decisiones individuales.')
                if compare:
                    rows=[]
                    for name,mdl in models.items():
                        p2=Pipeline([('prep',pre),('model',mdl)]).fit(Xtr,ytr).predict(Xte)
                        rows.append({'Algoritmo':name,'Accuracy':accuracy_score(yte,p2),'Balanced accuracy':balanced_accuracy_score(yte,p2),'F1 macro':f1_score(yte,p2,average='macro',zero_division=0)})
                    comparison_rows=rows
                    st.subheader('Comparación exploratoria de modelos');st.dataframe(pd.DataFrame(rows).round(3),hide_index=True)
                    metric='F1 macro' if kind=='Clasificación' else 'MAE'
                    st.plotly_chart(px.bar(pd.DataFrame(rows),x='Algoritmo',y=metric,title='Comparación de algoritmos: '+metric),use_container_width=True)
                if cv:
                    folds=min(3,int(ytr.value_counts().min()))
                    if folds>=2:
                        cvres=cross_validate(Pipeline([('prep',pre),('model',models[algorithm])]),Xtr,ytr,cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=int(seed)),scoring='f1_macro')
                        cv_details={'metrica':'F1 macro','valores':cvres['test_score'].tolist()}
                        st.caption(f'CV en entrenamiento, F1 macro: {cvres["test_score"].mean():.3f} ± {cvres["test_score"].std():.3f}')
                    else: st.warning('No hay suficientes casos por clase para validación cruzada.')
            elif kind=='Regresión':
                y=pd.to_numeric(work[target],errors='coerce'); mask=y.notna();X=X.loc[mask];y=y.loc[mask]
                if len(y)<20: raise ValueError('La regresión requiere al menos 20 valores numéricos.')
                models={'Regresión lineal':LinearRegression(),'Ridge':Ridge(),
                        'Árbol de regresión':DecisionTreeRegressor(max_depth=6,random_state=int(seed)),
                        'Random Forest':RandomForestRegressor(n_estimators=80,max_depth=10,random_state=int(seed),n_jobs=1)}
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_pct/100,random_state=int(seed))
                payload.update({'n_entrenamiento':len(Xtr),'n_prueba':len(Xte),'preprocesamiento_interno':'Imputación de numéricas por mediana, escalado estándar; imputación de categóricas por moda y codificación one-hot (según tipos presentes). Ajuste dentro del pipeline de entrenamiento.'})
                pipe=Pipeline([('prep',pre),('model',models[algorithm])]);pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
                mae=mean_absolute_error(yte,pred);rmse=np.sqrt(mean_squared_error(yte,pred));r2=r2_score(yte,pred)
                base=DummyRegressor(strategy='mean').fit(Xtr,ytr);base_mae=mean_absolute_error(yte,base.predict(Xte))
                result=f'Regresión {algorithm}: n prueba={len(yte)}, MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}; baseline MAE={base_mae:.3f}.'
                payload.update({'reales':yte.to_numpy().tolist(),'predicciones':np.asarray(pred).tolist(),'metricas':{'MAE':mae,'RMSE':rmse,'R2':r2,'Baseline MAE':base_mae}})
                st.subheader('Calificaciones reales y predichas')
                fig=px.scatter(x=yte.to_numpy(),y=pred,labels={'x':'Valor real','y':'Valor predicho'},title='Predicción frente a realidad')
                low=float(min(np.min(yte),np.min(pred)));high=float(max(np.max(yte),np.max(pred)))
                fig.add_trace(go.Scatter(x=[low,high],y=[low,high],mode='lines',name='Predicción ideal',line={'dash':'dash'}))
                st.plotly_chart(fig,use_container_width=True)
                residuals=yte.to_numpy()-pred
                st.plotly_chart(px.scatter(x=pred,y=residuals,labels={'x':'Valor predicho','y':'Residuo (real - predicho)'},title='Diagnóstico de residuos'),use_container_width=True)
                st.caption(f'Modelo de referencia (predice media del entrenamiento): MAE={base_mae:.3f}. Los residuos ayudan a identificar sesgos y errores grandes.')
                if compare:
                    rows=[]
                    for name,mdl in models.items():
                        p2=Pipeline([('prep',pre),('model',mdl)]).fit(Xtr,ytr).predict(Xte)
                        rows.append({'Algoritmo':name,'MAE':mean_absolute_error(yte,p2),'RMSE':np.sqrt(mean_squared_error(yte,p2)),'R²':r2_score(yte,p2)})
                    comparison_rows=rows
                    st.subheader('Comparación exploratoria de modelos');st.dataframe(pd.DataFrame(rows).round(3),hide_index=True)
                    metric='F1 macro' if kind=='Clasificación' else 'MAE'
                    st.plotly_chart(px.bar(pd.DataFrame(rows),x='Algoritmo',y=metric,title='Comparación de algoritmos: '+metric),use_container_width=True)
                if cv:
                    cvres=cross_validate(Pipeline([('prep',pre),('model',models[algorithm])]),Xtr,ytr,cv=KFold(n_splits=3,shuffle=True,random_state=int(seed)),scoring='neg_mean_absolute_error')
                    values=-cvres['test_score'];cv_details={'metrica':'MAE','valores':values.tolist()};st.caption(f'CV en entrenamiento, MAE: {values.mean():.3f} ± {values.std():.3f}')
            else:
                Xt=pre.fit_transform(X)
                models={'K-Means':KMeans(n_clusters=clusters,n_init=10,random_state=int(seed)),
                        'Jerárquico':AgglomerativeClustering(n_clusters=clusters),
                        'DBSCAN':DBSCAN(eps=.8,min_samples=5)}
                labels=models[algorithm].fit_predict(Xt)
                counts=pd.Series(labels).value_counts().sort_index()
                unique=set(labels)
                valid=(labels!=-1) if algorithm=='DBSCAN' else np.ones(len(labels),dtype=bool)
                score=silhouette_score(Xt[valid],labels[valid]) if len(set(labels[valid]))>1 and len(set(labels[valid]))<sum(valid) and sum(valid)<=2000 else np.nan
                result=f'Agrupamiento {algorithm}: n={len(Xt)}, grupos/etiquetas={len(unique)}, silhouette={score:.3f}.'
                payload.update({'etiquetas_grupo':pd.Series(labels).astype(int).tolist(),'conteos':counts.to_dict(),'metricas':{'Silhouette':float(score) if np.isfinite(score) else None}})
                st.subheader('Grupos identificados')
                st.dataframe(counts.rename_axis('Grupo').reset_index(name='Observaciones').assign(Porcentaje=lambda d:(100*d['Observaciones']/len(Xt)).round(1)),hide_index=True)
                st.subheader('Mapa de agrupamiento (proyección PCA)')
                coords=PCA(n_components=2,random_state=int(seed)).fit_transform(Xt)
                pca=PCA(n_components=2,random_state=int(seed)).fit(Xt)
                plotdf=pd.DataFrame({'Componente 1':coords[:,0],'Componente 2':coords[:,1],'Grupo':[str(x) if x!=-1 else 'Ruido (-1)' for x in labels]})
                fig=px.scatter(plotdf,x='Componente 1',y='Componente 2',color='Grupo',title='Cada punto representa una observación',opacity=.75)
                st.plotly_chart(fig,use_container_width=True)
                payload['pca_coordenadas']=coords.tolist()
                payload['pca_varianza']=float(sum(pca.explained_variance_ratio_))
                st.caption(f'PCA explica {100*sum(pca.explained_variance_ratio_):.1f}% de la varianza del espacio transformado en estos dos ejes. La separación visual es aproximada, no una validación de perfiles.')
                st.subheader('Caracterización de grupos')
                profiles=X.copy();profiles['Grupo']=labels
                if nums:
                    payload['perfiles']=profiles.groupby('Grupo')[nums].mean().reset_index().to_dict('records')
                    st.dataframe(profiles.groupby('Grupo')[nums].mean().round(2),use_container_width=True)
                    st.info('Descripción exploratoria: '+ ' | '.join(f'Grupo {g}: '+', '.join(f'{v}={row[v]:.2f}' for v in nums[:5]) for g,row in profiles.groupby('Grupo')[nums].mean().iterrows())+' . Son promedios de variables originales, no diagnósticos ni relaciones causales.')
                    chosen=st.selectbox('Variable para comparar perfiles',nums)
                    st.plotly_chart(px.box(profiles,x=profiles['Grupo'].astype(str),y=chosen,labels={'x':'Grupo'},title=f'{chosen} por grupo'),use_container_width=True)
                else: st.info('Selecciona al menos una variable numérica para comparar medias por grupo.')
                if compare and algorithm=='K-Means':
                    values=[]
                    for k in range(2,min(8,len(Xt)-1)+1):
                        lb=KMeans(n_clusters=k,n_init=10,random_state=int(seed)).fit_predict(Xt)
                        values.append({'k':k,'silhouette':silhouette_score(Xt,lb)})
                    payload['comparacion_k']=values
                    st.plotly_chart(px.line(pd.DataFrame(values),x='k',y='silhouette',markers=True,title='Comparación exploratoria de k'),use_container_width=True)
                    best=max(values,key=lambda row:row['silhouette'])
                    st.dataframe(pd.DataFrame(values).round(4),hide_index=True,use_container_width=True)
                    st.info(f'El mejor silhouette explorado se obtuvo con k={best["k"]} ({best["silhouette"]:.3f}); no demuestra que esos grupos sean perfiles pedagógicos válidos.')
                st.caption('Los grupos son patrones exploratorios, no diagnósticos de estudiantes. En DBSCAN, -1 identifica ruido.')
            payload['resultado']=result
            payload['comparacion']=comparison_rows
            payload['validacion_cruzada']=cv_details
            st.session_state.ml_payload=payload
            st.success(result)
            st.info(explain_ml(payload))
            st.session_state.ml_summary=result
            st.session_state.ml_details={'tarea':kind,'algoritmo':algorithm,'variables':predictors,'objetivo':target,'resultado':result}
            if kind=='Regresión':
                text_box(f'El error medio es **{mae:.2f}** unidades de la variable objetivo. R²={r2:.3f}. Compáralo con el baseline y revisa la dispersión y los residuos. No implica causalidad.')
            elif kind=='Clasificación':
                text_box(f'Accuracy **{acc:.3f}** frente a baseline **{baseline_acc:.3f}**; balanced accuracy **{bacc:.3f}**. Revisa las clases minoritarias antes de concluir que el clasificador es útil.')
            else:
                text_box(f'Se identificaron **{len(unique)} etiquetas** con silhouette **{score:.3f}**. Examina las medias por grupo y la proyección PCA; la estructura no equivale a perfiles pedagógicos validados.')
        except Exception as ex:
            st.error(f'No fue posible ejecutar el modelo: {ex}')

st.divider()
st.caption('LA-Lab v0.6.3 · Prototipo abierto para enseñanza e investigación · No utilizar para decisiones automatizadas sobre estudiantes. Identidad institucional: incorporación de emblemas sujeta a archivos oficiales y autorización de uso.')
