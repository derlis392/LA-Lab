"""LA-Lab v0.2 — Laboratorio abierto de Analítica del Aprendizaje."""
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
CASES = [
 ('Participación en aula virtual','Accesos, entregas y rendimiento','¿La actividad digital coincide con el desempeño?'),
 ('Seguimiento de entregas','Cumplimiento de tareas','¿Qué grupos requieren revisar las fechas de entrega?'),
 ('Asistencia y desempeño','Asistencia presencial y calificaciones','¿Cómo varía el desempeño según la asistencia?'),
 ('Evaluación formativa','Cuestionarios y progreso','¿Qué cambios aparecen entre evaluaciones?'),
 ('Trayectorias académicas','Créditos y avance','¿Qué perfiles de avance se observan?'),
 ('Foros de discusión','Interacciones y participación','¿Quién participa y con qué frecuencia?'),
 ('Uso de recursos digitales','Materiales y consultas','¿Qué recursos se consultan más?'),
 ('Cursos híbridos','Modalidad y participación','¿Existen diferencias descriptivas entre modalidades?'),
 ('Retroalimentación docente','Oportunidad de feedback','¿Cómo se distribuyen los tiempos de respuesta?'),
 ('Aprendizaje autorregulado','Planificación y actividades','¿Qué patrones de autorregulación se describen?'),
 ('Riesgo académico ilustrativo','Entregas, asistencia y calificación','¿Qué indicadores sugieren necesidad de acompañamiento?'),
 ('Brechas de acceso digital','Conectividad y actividad','¿Cómo se distribuye la actividad por conectividad?'),
 ('Interacciones con IA educativa','Consultas, revisión y reflexión','¿Qué tipos de uso de IA se registran?'),
 ('Satisfacción y experiencia','Encuestas y desempeño','¿Qué relación descriptiva hay entre experiencia y resultados?'),
]

@st.cache_data(show_spinner=False)
def make_case(index):
    rng = np.random.default_rng(20261012 + index)
    n = 150
    group = rng.choice(['A','B','C'], n)
    engagement = np.clip(rng.normal(0.55, 0.23, n),0,1)
    access = rng.poisson(4 + 45 * engagement)
    deliveries = rng.binomial(10, np.clip(.15+.75*engagement,0,1))
    attendance = np.clip(np.round(45 + 52*engagement + rng.normal(0,13,n)),0,100)
    grade = np.clip(np.round(35 + 45*engagement + .9*deliveries + rng.normal(0,13,n)),0,100)
    data = pd.DataFrame({'id_anonimo':[f'E{i:04d}' for i in range(1,n+1)], 'grupo':group,
        'accesos':access, 'entregas':deliveries, 'asistencia_pct':attendance.astype(int),
        'calificacion':grade.astype(int)})
    extras = {
      0:{}, 1:{'entregas_tardias':rng.binomial(5,1-engagement)},
      2:{'faltas':np.maximum(0,np.round((100-attendance)/10)).astype(int)},
      3:{'quiz_inicial':np.clip(grade-rng.integers(3,22,n),0,100).astype(int),'quiz_final':grade.astype(int)},
      4:{'creditos_aprobados':rng.integers(10,260,n),'semestre':rng.integers(1,10,n)},
      5:{'mensajes_foro':rng.poisson(1+12*engagement)},
      6:{'consultas_material':rng.poisson(2+15*engagement),'tipo_recurso':rng.choice(['Video','Lectura','Simulador'],n)},
      7:{'modalidad':rng.choice(['Presencial','Híbrida','Virtual'],n)},
      8:{'tiempo_feedback_h':rng.integers(1,120,n)},
      9:{'planificacion_1a5':rng.integers(1,6,n),'autoevaluacion_1a5':rng.integers(1,6,n)},
      10:{'tutorias_recibidas':rng.integers(0,5,n)},
      11:{'conectividad':rng.choice(['Alta','Media','Baja'],n)},
      12:{'consultas_ia':rng.poisson(1+12*engagement),'revision_fuentes':rng.choice(['Sí','No'],n)},
      13:{'satisfaccion_1a5':rng.integers(1,6,n)},
    }
    return data.assign(**extras.get(index,{}))

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
                return pd.read_csv(io.BytesIO(raw),sep=sep,engine='python',encoding=enc)
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
    st.caption('Laboratorio Virtual de Analítica del Aprendizaje · v0.2')
    mode = st.radio('Modalidad', ['Casos guiados','Datos propios'],key='mode')
    idx = None
    if mode == 'Casos guiados':
        idx = st.selectbox('Caso de estudio',range(len(CASES)),format_func=lambda i:f'{i+1:02d}. {CASES[i][0]}')
    else:
        upload = st.file_uploader('Subir datos (máximo 10 MB)',type=['csv','tsv','txt','xlsx','xls','json','parquet'])
    page = st.radio('Módulo',['Inicio','1. Datos','2. Limpieza','3. Indicadores','4. Visualización','5. Interpretación','6. Informe'])
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
    df_original = make_case(idx)
    title, description, question = CASES[idx]
    st.caption('CASO SINTÉTICO PARA DOCENCIA · no es un conjunto de datos real ni una muestra de una institución')
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

source_key = f'{mode}-{idx if idx is not None else ""}-{upload.name if mode=="Datos propios" and upload else ""}-{upload.size if mode=="Datos propios" and upload else ""}'
if st.session_state.get('source_key')!=source_key:
    st.session_state.source_key=source_key
    st.session_state.cleaned=None
    st.session_state.last_cleaning='No se han aplicado transformaciones.'
df = st.session_state.cleaned if st.session_state.cleaned is not None else df_original

if page=='Inicio':
    st.header('Bienvenido al laboratorio')
    st.write('Selecciona un caso guiado o carga una base propia. Recorre los módulos para preparar, analizar, visualizar e interpretar los datos.')
    a,b=st.columns(2)
    with a:
        st.subheader('📚 Laboratorio guiado')
        st.write('14 casos temáticos sintéticos, con preguntas de investigación y actividades orientadas a la analítica educativa.')
    with b:
        st.subheader('📂 Laboratorio libre')
        st.write('Carga CSV, TSV, TXT delimitado, Excel, JSON de registros o Parquet y analiza variables mediante controles gráficos.')
    st.markdown('##### Procedencia y alcance')
    st.info('Los 14 casos integrados son **sintéticos**, generados para enseñar conceptos presentes en la literatura de Learning Analytics. No se presentan como bases reales ni reproducen resultados publicados. Se incluyen enlaces a repositorios abiertos verificables en la sección de fuentes.')
    st.markdown('##### Fuentes abiertas para ampliar el laboratorio')
    st.markdown('- [Open University Learning Analytics Dataset (OULAD)](https://doi.org/10.24432/C5KK69) — datos reales de cursos y actividad virtual, CC BY 4.0.\n- [Student Performance (UCI)](https://doi.org/10.24432/C5TG7T) — desempeño académico de secundaria, CC BY 4.0.\n- [Predict Students’ Dropout and Academic Success (UCI)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) — trayectoria y resultados de educación superior; consultar condiciones antes de reutilizar.')
    st.caption('Los datos reales no están incorporados ni descargados automáticamente en esta versión.')
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
    report=f'''LA-LAB v0.2 | INFORME EXPLORATORIO\nCaso: {title}\nProcedencia: {"SINTÉTICA, generada con fines docentes" if mode=="Casos guiados" else "Archivo aportado por el usuario"}\nRegistros: {len(df)}\nVariables: {len(df.columns)}\nFaltantes: {int(df.isna().sum().sum())}\nDuplicados completos: {int(df.duplicated().sum())}\nLimpieza: {st.session_state.last_cleaning}\n\nVARIABLES:\n{', '.join(map(str,df.columns))}\n\nESTADÍSTICAS NUMÉRICAS:\n{df.describe().to_string() if len(df.select_dtypes(include="number").columns) else "No hay variables numéricas"}\n\nNOTAS:\nLas asociaciones no implican causalidad. Los indicadores requieren validación contextual.\n'''
    st.download_button('Descargar informe (.txt)',report.encode('utf-8'),'informe_la_lab.txt','text/plain')
    st.download_button('Descargar datos actuales (.csv)',csv_bytes(df),'datos_la_lab.csv','text/csv')
    text_box('El informe contiene el resumen de la base, los procedimientos de limpieza y las estadísticas numéricas. No constituye una evaluación individual ni una recomendación automática sobre estudiantes.')

st.divider()
st.caption('LA-Lab v0.2 · Prototipo abierto para enseñanza e investigación · No utilizar para decisiones automatizadas sobre estudiantes. Identidad institucional: incorporación de emblemas sujeta a archivos oficiales y autorización de uso.')
