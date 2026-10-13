"""LA-Lab v0.3 — Laboratorio abierto de Analítica del Aprendizaje."""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='LA-Lab | Analítica del Aprendizaje', page_icon='🎓', layout='wide')
ROOT = Path(__file__).parent
MAX_BYTES = 10 * 1024 * 1024
# Catorce prácticas sobre tres fuentes originales; no son 14 datasets independientes.
SOURCES = {
 'student_mat': {'name':'Student Performance — Matemáticas', 'uci_id':320, 'url':'https://archive.ics.uci.edu/dataset/320/student+performance', 'article':'Cortez, P. & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance.', 'article_url':'https://www.openml.org/search?type=data&id=42352', 'license':'CC BY 4.0', 'target':'G3'},
 'student_por': {'name':'Student Performance — Portugués', 'uci_id':320, 'url':'https://archive.ics.uci.edu/dataset/320/student+performance', 'article':'Cortez, P. & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance.', 'article_url':'https://www.openml.org/search?type=data&id=42352', 'license':'CC BY 4.0', 'target':'G3'},
 'dropout': {'name':'Predict Students Dropout and Academic Success', 'uci_id':697, 'url':'https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success', 'article':'Realinho, V., Machado, J., Baptista, L. & Martins, M. (2022). Predicting Student Dropout and Academic Success. Data, 7(11), 146.', 'article_url':'https://doi.org/10.3390/data7110146', 'license':'CC BY 4.0', 'target':'Target'},
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
 ('Actividad virtual y resultados','oulad','¿Cómo relacionar interacciones VLE y resultados?','final_result'),
 ('Evaluaciones en educación a distancia','oulad','¿Qué patrones presentan las evaluaciones y sus resultados?','final_result'),
 ('Participación y permanencia','oulad','¿Qué indicadores se relacionan con permanencia?','final_result'),
]

@st.cache_data(show_spinner=False, ttl=86400)
def load_original(source_id):
    # Descarga bajo demanda en el servidor Streamlit; nunca se sustituyen datos reales por sintéticos.
    from ucimlrepo import fetch_ucirepo
    source = SOURCES[source_id]
    if source_id == 'oulad':
        raise ValueError('OULAD requiere descargar y preparar sus tablas originales. Utiliza la modalidad Datos propios para cargar una tabla CSV original; el enlace oficial está en la ficha del caso.')
    if source_id.startswith('student_'):
        import urllib.request, zipfile
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
    ds = fetch_ucirepo(id=source['uci_id'])
    df = pd.concat([ds.data.features.reset_index(drop=True),ds.data.targets.reset_index(drop=True)],axis=1)
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
    st.caption('Laboratorio Virtual de Analítica del Aprendizaje · v0.3')
    mode = st.radio('Modalidad', ['Casos guiados','Datos propios'],key='mode')
    idx = None
    if mode == 'Casos guiados':
        idx = st.selectbox('Caso de estudio',range(len(CASES)),format_func=lambda i:f'{i+1:02d}. {CASES[i][0]}')
    else:
        upload = st.file_uploader('Subir datos (máximo 10 MB)',type=['csv','tsv','txt','xlsx','xls','json','parquet'])
    page = st.radio('Módulo',['Inicio','1. Datos','2. Limpieza','3. Indicadores','4. Visualización','5. Interpretación','6. Informe','7. Minería de datos'])
    st.warning('Utiliza datos sintéticos o autorizados y desidentificados. No cargues datos personales sensibles.')

header_left, header_right = st.columns([4,1])
with header_left:
    st.title('🎓 LA-Lab')
    st.caption('Tecnologías Inteligentes para la Transformación Digital de los Ecosistemas Educativos')
    st.caption('Dr. Derlis Hernández Lara · CUAED · Universidad Nacional Autónoma de México')
with header_right:
    logos = ROOT/'assets'
    logos_found = [(label, logos/file) for label,file in [('UNAM','unam.png'),('CUAED','cuaed.png')] if (logos/file).exists()]
    if logos_found:
        cols = st.columns(len(logos_found))
        for col,(label,p) in zip(cols,logos_found): col.image(str(p),caption=label,use_container_width=True)
    else:
        st.markdown('**UNAM · CUAED**')
        st.caption('Logotipos oficiales pendientes de incorporar')

if mode=='Casos guiados':
    title, source_id, question, suggested_target = CASES[idx]
    source = SOURCES[source_id]
    description = source['name']
    try:
        df_original = load_original(source_id)
    except Exception as ex:
        df_original = None
        st.info(str(ex))
    st.caption('FUENTE ORIGINAL: '+source['name']+' · '+source['license'])
    with st.expander('Ficha científica, artículo y descarga del conjunto original',expanded=True):
        st.write('**Pregunta:** '+question)
        st.write('**Fuente:** '+source['name'])
        st.link_button('Abrir fuente y descargar datos originales',source['url'])
        st.write('**Artículo asociado:** '+source['article'])
        st.link_button('Consultar publicación de referencia',source['article_url'])
        st.caption('Los artículos documentan los datos o su análisis; los resultados de LA-Lab son nuevos y no equivalen automáticamente a una reproducción.')
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
df = st.session_state.cleaned if st.session_state.cleaned is not None else df_original

if page=='Inicio':
    st.header('Bienvenido al laboratorio')
    st.write('Selecciona un caso guiado o carga una base propia. Recorre los módulos para preparar, analizar, visualizar e interpretar los datos.')
    a,b=st.columns(2)
    with a:
        st.subheader('📚 Laboratorio guiado')
        st.write('14 prácticas sobre fuentes académicas originales, con preguntas de investigación y actividades orientadas a la analítica educativa.')
    with b:
        st.subheader('📂 Laboratorio libre')
        st.write('Carga CSV, TSV, TXT delimitado, Excel, JSON de registros o Parquet y analiza variables mediante controles gráficos.')
    st.markdown('##### Procedencia y alcance')
    st.info('Los 14 casos son **prácticas sobre tres colecciones de datos reales**. UCI se consulta desde sus fuentes oficiales; OULAD requiere descarga y preparación manual. Los artículos citados no constituyen una validación automática de los resultados obtenidos aquí.')
    st.markdown('##### Fuentes abiertas para ampliar el laboratorio')
    st.markdown('- [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset) — datos reales de cursos y actividad virtual, CC BY 4.0.\n- [Student Performance (UCI)](https://doi.org/10.24432/C5TG7T) — desempeño académico de secundaria, CC BY 4.0.\n- [Predict Students’ Dropout and Academic Success (UCI)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) — trayectoria y resultados de educación superior; consultar condiciones antes de reutilizar.')
    st.caption('Los casos UCI se descargan de la fuente original cuando el servidor tiene acceso. OULAD requiere descargar y preparar sus tablas oficiales. No se generan datos ficticios como sustitutos.')
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
        st.session_state.last_cleaning=f'Filas antes: {before}; después: {len(cleaned)}. Duplicados: {remove}. Faltantes: {missing}.'
        st.success(st.session_state.last_cleaning)
        st.rerun()
    if st.button('Restablecer datos originales'):
        st.session_state.cleaned=None
        st.session_state.last_cleaning='Datos originales restablecidos.'
        st.rerun()
    st.download_button('Descargar tabla actual (CSV)',csv_bytes(df),'la_lab_datos_limpios.csv','text/csv')
    text_box(f'El conjunto actual tiene **{len(df)} filas**, **{int(df.isna().sum().sum())} valores faltantes** y **{int(df.duplicated().sum())} duplicados completos**. {st.session_state.last_cleaning} Antes de imputar, revisa si los datos faltan sistemáticamente; la imputación puede alterar la distribución.')
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
    text_box(summarize_numeric(df[var])+' Estas cifras describen la muestra; su relevancia educativa depende de la escala, el periodo y el contexto.')
elif page=='4. Visualización':
    st.header('4. Visualización exploratoria')
    numeric=df.select_dtypes(include='number').columns.tolist()
    cats=df.select_dtypes(exclude='number').columns.tolist()
    kinds=['Histograma','Diagrama de dispersión','Diagrama de caja','Frecuencia de categorías','Matriz de correlación']
    kind=st.selectbox('Tipo de gráfico',kinds)
    if kind=='Histograma' and numeric:
        x=st.selectbox('Variable',numeric); fig=px.histogram(df,x=x,nbins=25)
        st.plotly_chart(fig,use_container_width=True)
        text_box(f'El histograma muestra la distribución de **{x}**. {summarize_numeric(df[x])} Examina concentración, asimetría y posibles valores extremos.')
    elif kind=='Diagrama de dispersión' and len(numeric)>=2:
        x=st.selectbox('Eje X',numeric); y=st.selectbox('Eje Y',[c for c in numeric if c!=x]); fig=px.scatter(df,x=x,y=y,opacity=.7)
        st.plotly_chart(fig,use_container_width=True)
        pair=df[[x,y]].dropna()
        r=pair[x].corr(pair[y]) if len(pair)>2 and pair[x].nunique()>1 and pair[y].nunique()>1 else np.nan
        msg=f'La correlación de Pearson entre **{x}** y **{y}** es **{r:.2f}**.' if pd.notna(r) else 'No es posible calcular una correlación válida.'
        st.session_state.plot_summary=msg
        text_box(msg+' La asociación no demuestra causalidad; evalúa valores extremos, linealidad y posibles factores de confusión.')
    elif kind=='Diagrama de caja' and numeric:
        y=st.selectbox('Variable',numeric); group=st.selectbox('Agrupar por (opcional)',['Sin grupos']+cats)
        fig=px.box(df,y=y,x=group if group!='Sin grupos' else None)
        st.plotly_chart(fig,use_container_width=True)
        text_box(f'El diagrama de caja permite comparar medianas, dispersión y observaciones extremas de **{y}**'+(f' por **{group}**.' if group!='Sin grupos' else '.')+' Las diferencias descriptivas no implican efectos causales.')
    elif kind=='Frecuencia de categorías' and cats:
        x=st.selectbox('Variable categórica',cats)
        counts=df[x].astype('string').fillna('(Faltante)').value_counts().head(20).rename_axis(x).reset_index(name='frecuencia')
        st.plotly_chart(px.bar(counts,x=x,y='frecuencia'),use_container_width=True)
        text_box(f'La categoría más frecuente de **{x}** es **{counts.iloc[0,0]}**, con **{counts.iloc[0,1]} registros**. Se muestran como máximo 20 categorías.')
    elif kind=='Matriz de correlación' and len(numeric)>=2:
        corr=df[numeric].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale='RdBu_r'),use_container_width=True)
        st.session_state.plot_summary='Matriz de correlación calculada para '+str(len(numeric))+' variables numéricas.'
        text_box('La matriz muestra asociaciones lineales entre variables numéricas. Correlaciones elevadas pueden reflejar variables redundantes, relaciones mecánicas o patrones que requieren investigación adicional.')
    else:
        st.warning('El archivo no tiene suficientes variables compatibles con este tipo de gráfico.')
elif page=='5. Interpretación':
    st.header('5. Interpretación y decisiones educativas')
    st.write('**Pregunta orientadora:** '+question)
    text_box(interpret_dataset(df))
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
    st.header('6. Informe descargable')
    st.write('Síntesis reproducible de la base actual. Las conclusiones automáticas son descriptivas y deben revisarse antes de su uso académico.')
    st.info(interpret_dataset(df))
    report=f'''LA-LAB v0.3 | INFORME EXPLORATORIO\nCaso: {title}\nProcedencia: {"Conjunto real: "+description if mode=="Casos guiados" else "Archivo aportado por el usuario"}\nRegistros: {len(df)}\nVariables: {len(df.columns)}\nFaltantes: {int(df.isna().sum().sum())}\nDuplicados completos: {int(df.duplicated().sum())}\nLimpieza: {st.session_state.last_cleaning}\nGráfico: {st.session_state.get('plot_summary','No evaluado')}\nMinería de datos: {st.session_state.get('ml_summary','No evaluado')}\n\nVARIABLES:\n{', '.join(map(str,df.columns))}\n\nESTADÍSTICAS NUMÉRICAS:\n{df.describe().to_string() if len(df.select_dtypes(include="number").columns) else "No hay variables numéricas"}\n\nNOTAS:\nLas asociaciones no implican causalidad. Los indicadores requieren validación contextual.\n'''
    st.download_button('Descargar informe (.txt)',report.encode('utf-8'),'informe_la_lab.txt','text/plain')
    st.download_button('Descargar datos actuales (.csv)',csv_bytes(df),'datos_la_lab.csv','text/csv')
    st.download_button('Descargar informe integrado (.txt)',report.encode('utf-8'),'informe_integrado_la_lab.txt','text/plain')
    text_box('El informe contiene el resumen de la base, los procedimientos de limpieza y las estadísticas numéricas. No constituye una evaluación individual ni una recomendación automática sobre estudiantes.')


elif page=='7. Minería de datos':
    st.header('7. Minería de datos y aprendizaje automático')
    st.write('Exploración reproducible para enseñanza. No utilizar modelos para decisiones automatizadas sobre estudiantes.')
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
    kind=st.selectbox('Tipo de tarea',['Clasificación','Regresión','Agrupamiento'])
    options=list(df.columns)
    if kind!='Agrupamiento':
        target=st.selectbox('Variable objetivo',options,index=options.index(suggested_target) if suggested_target in options else len(options)-1)
        predictors=st.multiselect('Variables predictoras',[c for c in options if c!=target],default=[c for c in options if c!=target][:min(12,len(options)-1)])
    else:
        target=None
        predictors=st.multiselect('Variables para agrupar',options,default=df.select_dtypes(include='number').columns.tolist()[:8])
    choices={'Clasificación':['Regresión logística','Árbol de decisión','Random Forest','SVM','KNN','Gradient Boosting'], 'Regresión':['Regresión lineal','Ridge','Árbol de regresión','Random Forest'], 'Agrupamiento':['K-Means','Jerárquico','DBSCAN']}
    algorithm=st.selectbox('Algoritmo',choices[kind])
    seed=st.number_input('Semilla aleatoria',0,100000,42)
    if kind!='Agrupamiento': test_pct=st.slider('Porcentaje de prueba',15,40,25,5)
    else: clusters=st.slider('Número de grupos (si aplica)',2,8,3)
    st.caption('Se aplica imputación y codificación dentro del pipeline. Se limita a 12 predictores recomendados y hasta 5 000 filas para proteger el servidor gratuito.')
    if st.button('Ejecutar análisis',type='primary'):
        try:
            if not predictors: raise ValueError('Selecciona al menos una variable predictora.')
            work=df[predictors+([target] if target else [])].copy()
            if target: work=work.dropna(subset=[target])
            if len(work)>5000: work=work.sample(5000,random_state=int(seed))
            if len(work)<20: raise ValueError('Se necesitan al menos 20 registros válidos.')
            X=work[predictors].copy()
            for c in X.columns:
                if pd.api.types.is_datetime64_any_dtype(X[c]): X[c]=X[c].astype(str)
            nums=X.select_dtypes(include='number').columns.tolist()
            cats=[c for c in predictors if c not in nums]
            transformers=[]
            if nums: transformers.append(('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums))
            if cats: transformers.append(('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats))
            pre=ColumnTransformer(transformers)
            if kind=='Clasificación':
                y=work[target].astype(str)
                if y.nunique()<2 or y.nunique()>30: raise ValueError('La clasificación requiere entre 2 y 30 categorías en la variable objetivo.')
                models={'Regresión logística':LogisticRegression(max_iter=500),'Árbol de decisión':DecisionTreeClassifier(max_depth=6,random_state=int(seed)),'Random Forest':RandomForestClassifier(n_estimators=80,max_depth=10,random_state=int(seed),n_jobs=1),'SVM':SVC(),'KNN':KNeighborsClassifier(),'Gradient Boosting':GradientBoostingClassifier(random_state=int(seed))}
                model=models[algorithm]
                strat=y if y.value_counts().min()>=2 else None
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_pct/100,random_state=int(seed),stratify=strat)
                pipe=Pipeline([('prep',pre),('model',model)]);pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
                result=f'Clasificación {algorithm}: n prueba={len(yte)}, accuracy={accuracy_score(yte,pred):.3f}, balanced accuracy={balanced_accuracy_score(yte,pred):.3f}, F1 macro={f1_score(yte,pred,average="macro",zero_division=0):.3f}.'
                st.dataframe(pd.DataFrame(confusion_matrix(yte,pred,labels=pipe.classes_),index=pipe.classes_,columns=pipe.classes_))
                st.write('Distribución de clases de prueba:',yte.value_counts().to_dict())
            elif kind=='Regresión':
                y=pd.to_numeric(work[target],errors='coerce')
                mask=y.notna();X=X.loc[mask];y=y.loc[mask]
                if len(y)<20: raise ValueError('La regresión requiere una variable objetivo numérica con al menos 20 valores.')
                models={'Regresión lineal':LinearRegression(),'Ridge':Ridge(),'Árbol de regresión':DecisionTreeRegressor(max_depth=6,random_state=int(seed)),'Random Forest':RandomForestRegressor(n_estimators=80,max_depth=10,random_state=int(seed),n_jobs=1)}
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_pct/100,random_state=int(seed))
                pipe=Pipeline([('prep',pre),('model',models[algorithm])]);pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
                result=f'Regresión {algorithm}: n prueba={len(yte)}, MAE={mean_absolute_error(yte,pred):.3f}, RMSE={np.sqrt(mean_squared_error(yte,pred)):.3f}, R²={r2_score(yte,pred):.3f}.'
                st.plotly_chart(px.scatter(x=yte,y=pred,labels={'x':'Real','y':'Predicho'}),use_container_width=True)
            else:
                Xt=pre.fit_transform(X)
                models={'K-Means':KMeans(n_clusters=clusters,n_init=10,random_state=int(seed)),'Jerárquico':AgglomerativeClustering(n_clusters=clusters),'DBSCAN':DBSCAN(eps=.8,min_samples=5)}
                labels=models[algorithm].fit_predict(Xt)
                counts=pd.Series(labels).value_counts().sort_index()
                score=silhouette_score(Xt,labels) if 1<len(set(labels))<len(labels) and len(Xt)<=2000 else np.nan
                result=f'Agrupamiento {algorithm}: n={len(Xt)}, grupos/etiquetas={len(set(labels))}, silhouette={score:.3f} (si es calculable).'
                st.dataframe(counts.rename_axis('Grupo').reset_index(name='Observaciones'),hide_index=True)
                st.caption('En DBSCAN, la etiqueta -1 representa ruido. Silhouette es exploratorio y no valida perfiles pedagógicos.')
            st.success(result)
            st.session_state.ml_summary=result
            text_box('**Hallazgo técnico:** '+result+' Es un resultado de esta ejecución, no una reproducción del artículo de referencia. Revisa sesgos, variables disponibles al momento de predicción, representatividad y posibles fugas de información.')
        except Exception as ex:
            st.error(f'No fue posible ejecutar el modelo: {ex}')

st.divider()
st.caption('LA-Lab v0.3 · Prototipo abierto para enseñanza e investigación · No utilizar para decisiones automatizadas sobre estudiantes. Identidad institucional: incorporación de emblemas sujeta a archivos oficiales y autorización de uso.')
