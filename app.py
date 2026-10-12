import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='LA-Lab | Analítica del Aprendizaje', page_icon='🎓', layout='wide')
st.title('🎓 LA-Lab')
st.caption('Laboratorio Virtual de Analítica del Aprendizaje · Prototipo abierto v0.1')

@st.cache_data
def example_data():
    rng = np.random.default_rng(42)
    n = 120
    entregas = rng.integers(2, 11, n)
    accesos = rng.integers(1, 55, n)
    notas = np.clip(np.round(35 + 4.2 * entregas + 0.27 * accesos + rng.normal(0, 11, n)), 0, 100)
    return pd.DataFrame({'id_anonimo': [f'E{i:03d}' for i in range(1, n+1)],
                         'grupo': rng.choice(['A','B','C'], n),
                         'accesos': accesos, 'entregas': entregas,
                         'calificacion': notas.astype(int)})

st.sidebar.header('Navegación')
modulo = st.sidebar.radio('Módulo', ['Inicio', '1. Datos', '2. Limpieza', '3. Indicadores', '4. Visualización', '5. Interpretación'])
st.sidebar.info('Usa exclusivamente datos sintéticos o autorizados y anonimizados. No cargues información personal sensible.')

st.sidebar.subheader('Fuente de datos')
archivo = st.sidebar.file_uploader('CSV o Excel (máx. 10 MB)', type=['csv','xlsx'])
if archivo is not None:
    if archivo.size > 10 * 1024 * 1024:
        st.error('El archivo supera 10 MB.'); st.stop()
    try:
        if archivo.name.lower().endswith('.csv'):
            try:
                datos = pd.read_csv(archivo)
            except UnicodeDecodeError:
                archivo.seek(0)
                datos = pd.read_csv(archivo, encoding='latin-1')
        else:
            datos = pd.read_excel(archivo, engine='openpyxl')
        if len(datos.columns)>100 or len(datos)>100000:
            st.error('Límite de demostración: 100 columnas y 100 000 filas.'); st.stop()
    except Exception:
        st.error('No fue posible leer el archivo. Verifica su formato.'); st.stop()
else:
    datos = example_data()

if 'data_signature' not in st.session_state or st.session_state.data_signature != (archivo.name, archivo.size) if archivo else st.session_state.get('data_signature') != ('demo', 0):
    st.session_state.data_signature = (archivo.name, archivo.size) if archivo else ('demo', 0)
    st.session_state.clean_data = None

if modulo == 'Inicio':
    st.subheader('Bienvenido al laboratorio')
    st.write('Explora datos educativos, depura registros, calcula indicadores y visualiza patrones sin programar.')
    st.info('Este es un prototipo inicial. Los datos precargados son sintéticos y no representan estudiantes reales.')
    st.write('Selecciona un módulo en la barra lateral para comenzar.')

elif modulo == '1. Datos':
    st.subheader('Exploración de datos')
    c1,c2,c3=st.columns(3)
    c1.metric('Registros', len(datos)); c2.metric('Variables', len(datos.columns)); c3.metric('Celdas vacías', int(datos.isna().sum().sum()))
    st.dataframe(datos.head(100), use_container_width=True)
    st.write('Tipos de variables'); st.dataframe(pd.DataFrame({'variable':datos.columns,'tipo':datos.dtypes.astype(str).values}))

elif modulo == '2. Limpieza':
    st.subheader('Preparación y limpieza')
    st.write('Valores faltantes por variable'); st.dataframe(datos.isna().sum().rename('faltantes'))
    st.write('Filas duplicadas:', int(datos.duplicated().sum()))
    quitar_duplicados = st.checkbox('Eliminar filas duplicadas', value=True)
    tratar_vacios = st.selectbox('Tratamiento de valores faltantes', ['Conservar', 'Eliminar filas incompletas', 'Imputar numéricas con mediana'])
    if st.button('Aplicar transformaciones'):
        limpio = datos.copy()
        if quitar_duplicados: limpio = limpio.drop_duplicates()
        if tratar_vacios == 'Eliminar filas incompletas': limpio = limpio.dropna()
        elif tratar_vacios == 'Imputar numéricas con mediana':
            cols = limpio.select_dtypes(include='number').columns
            limpio[cols] = limpio[cols].fillna(limpio[cols].median())
        st.session_state.clean_data = limpio
    if st.session_state.clean_data is not None:
        st.success('Transformación aplicada. Revisa el resultado antes de usarlo.')
        st.dataframe(st.session_state.clean_data.head(50))
        st.download_button('Descargar CSV depurado', st.session_state.clean_data.to_csv(index=False).encode('utf-8-sig'), 'datos_depurados.csv', 'text/csv')

elif modulo == '3. Indicadores':
    st.subheader('Estadística e indicadores')
    df = st.session_state.clean_data if st.session_state.clean_data is not None else datos
    cols = df.select_dtypes(include='number').columns.tolist()
    if not cols: st.warning('Se necesitan variables numéricas.')
    else:
        var = st.selectbox('Variable numérica', cols)
        s = df[var].dropna()
        a,b,c,d = st.columns(4)
        a.metric('Media', f'{s.mean():.2f}' if len(s) else 'N/D')
        b.metric('Mediana', f'{s.median():.2f}' if len(s) else 'N/D')
        c.metric('Desv. estándar', f'{s.std():.2f}' if len(s)>1 else 'N/D')
        d.metric('Datos válidos', len(s))
        st.dataframe(df[cols].describe().T)

elif modulo == '4. Visualización':
    st.subheader('Visualización interactiva')
    df = st.session_state.clean_data if st.session_state.clean_data is not None else datos
    nums = df.select_dtypes(include='number').columns.tolist()
    tipo = st.selectbox('Tipo de gráfico', ['Histograma', 'Dispersión', 'Caja'])
    if not nums: st.warning('Se necesitan variables numéricas.')
    else:
        x = st.selectbox('Variable X', nums)
        if tipo == 'Histograma': fig = px.histogram(df, x=x)
        elif tipo == 'Caja': fig = px.box(df, y=x)
        else:
            y = st.selectbox('Variable Y', nums, index=min(1,len(nums)-1))
            fig = px.scatter(df, x=x, y=y)
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Una asociación visual no demuestra causalidad.')

elif modulo == '5. Interpretación':
    st.subheader('Práctica de interpretación')
    st.write('Un grupo registra menos accesos a la plataforma que otro. ¿Podemos concluir que aprendió menos?')
    respuesta = st.radio('Selecciona tu respuesta', ['Seleccionar...', 'Sí, necesariamente', 'No; debemos contrastar otras evidencias'], index=0)
    if st.button('Verificar'):
        if respuesta == 'Seleccionar...': st.warning('Selecciona una respuesta.')
        elif respuesta.startswith('No;'): st.success('Correcto. Los accesos no miden directamente el aprendizaje.')
        else: st.error('Revisa la inferencia: actividad digital y aprendizaje no son equivalentes.')

st.divider()
st.caption('LA-Lab v0.1 · Prototipo educativo · No usar resultados para decisiones automatizadas sobre estudiantes.')
