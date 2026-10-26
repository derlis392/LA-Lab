"""Módulo introductorio: paradoja del cumpleaños y analítica educativa exploratoria.
Los datos sintéticos son únicamente para pruebas y no representan estudiantes reales.
"""
import io
import re
import math
import unicodedata
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

MONTHS = {'enero':1,'febrero':2,'marzo':3,'abril':4,'mayo':5,'junio':6,'julio':7,'agosto':8,'septiembre':9,'octubre':10,'noviembre':11,'diciembre':12}
MONTH_NAMES = list(MONTHS)
DAYS = [31,29,31,30,31,30,31,31,30,31,30,31]
COLS = ['mes','dia','estimacion','promedio','estudio','sueno','traslado','modalidad','confianza','ia']
LABELS = {'promedio':'Promedio académico (0–10)','estudio':'Estudio independiente (h/semana)','sueno':'Sueño (h/noche)','traslado':'Traslado (min)','confianza':'Confianza en estadística (1–5)'}

def identify_columns(df):
    """Identifica encabezados numerados del Google Forms original y permite encabezados breves."""
    mapping = {}
    for c in df.columns:
        s = str(c).strip()
        m = re.match(r'^(10|[1-9])\s*[.\-)]\s*', s)
        if m:
            mapping[COLS[int(m.group(1))-1]] = c
            continue
        norm = ''.join(x for x in unicodedata.normalize('NFKD',s.lower()) if not unicodedata.combining(x))
        for key in COLS:
            if norm == key:
                mapping[key] = c
    return mapping

def prepare(df):
    mapping = identify_columns(df)
    if 'mes' not in mapping or 'dia' not in mapping:
        raise ValueError('No se identificaron las preguntas 1 (mes) y 2 (día). Utiliza el Excel exportado del formulario o encabezados «mes» y «dia».')
    out = pd.DataFrame(index=df.index)
    for key in COLS:
        out[key] = df[mapping[key]] if key in mapping else np.nan
    out['mes'] = out['mes'].astype('string').str.strip().str.lower()
    out['mes_num'] = out['mes'].map(MONTHS)
    out['dia'] = pd.to_numeric(out['dia'],errors='coerce')
    valid = out['mes_num'].notna() & out['dia'].notna() & (out['dia'] % 1 == 0)
    valid &= out.apply(lambda r: bool(pd.notna(r['mes_num']) and pd.notna(r['dia']) and 1 <= r['dia'] <= DAYS[int(r['mes_num'])-1]),axis=1)
    out['cumple_valido'] = valid
    out['fecha_cumple'] = out.apply(lambda r: f"{int(r['dia']):02d}/{int(r['mes_num']):02d}" if r['cumple_valido'] else None,axis=1)
    for key in ('promedio','estudio','sueno','traslado','confianza'):
        out[key] = pd.to_numeric(out[key].astype('string').str.replace(',','.',regex=False),errors='coerce')
    limits = {'promedio':(0,10),'estudio':(0,168),'sueno':(0,24),'traslado':(0,300),'confianza':(1,5)}
    for key,(lo,hi) in limits.items():
        out.loc[~out[key].between(lo,hi),key] = np.nan
    return out

def birthday_probability(n,days=365):
    if n < 2: return 0.0
    if n > days: return 1.0
    return 1.0 - math.exp(sum(math.log1p(-k/days) for k in range(n)))

def empirical(df,condition,subset=None):
    data = df if subset is None else df.loc[subset]
    valid = data['promedio'].notna()
    denom = int(valid.sum())
    return (int(condition.loc[data.index][valid].sum()),denom) if denom else (0,0)

def render():
    st.header('🎲 Motivación | De la encuesta a la analítica del aprendizaje')
    st.write('Actividad introductoria: recopilar datos, verificar su calidad, explorar distribuciones y contrastar una predicción probabilística con un evento observado.')
    st.info('**Modo de demostración:** los datos simulados no corresponden a estudiantes reales. La probabilidad teórica no se valida con un solo grupo; se ilustra mediante simulación de grupos independientes.')
    source = st.radio('Fuente de datos',['Ejemplo sintético (60 respuestas)','Cargar Excel/CSV de Google Forms'],horizontal=True,key='mot_source')
    if source.startswith('Ejemplo'):
        path = __import__('pathlib').Path(__file__).parent/'data'/'respuestas_sinteticas_60.xlsx'
        raw = pd.read_excel(path)
        st.download_button('Descargar Excel sintético para prácticas',path.read_bytes(),file_name='respuestas_sinteticas_60.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        uploaded = st.file_uploader('Selecciona el archivo exportado del formulario (máximo 10 MB)',type=['xlsx','csv'],key='mot_upload')
        if uploaded is None:
            st.caption('La pestaña funciona también sin respuestas reales; selecciona el ejemplo sintético para explorarla.')
            return
        if uploaded.size > 10*1024*1024:
            st.error('El archivo excede 10 MB.');return
        try:
            raw = pd.read_excel(uploaded) if uploaded.name.lower().endswith('.xlsx') else pd.read_csv(uploaded,encoding='utf-8-sig')
        except Exception as exc:
            st.error(f'No se pudo leer el archivo: {exc}');return
    if len(raw)>10000:
        st.error('Esta actividad admite hasta 10 000 registros.');return
    try:
        df = prepare(raw)
    except ValueError as exc:
        st.error(str(exc));return
    valid = df.loc[df['cumple_valido']]
    n = len(valid)
    freq = valid['fecha_cumple'].value_counts()
    matches = freq[freq>=2]
    participants = int(matches.sum())
    groups = int(len(matches))
    probability = birthday_probability(n)
    st.subheader('1. Calidad de los datos y paradoja del cumpleaños')
    if n != len(df):
        st.warning(f'{len(df)-n} respuesta(s) sin fecha de cumpleaños válida. Se excluyen del cálculo de coincidencias.')
    include_leap = st.checkbox('Incluir el 29 de febrero en el análisis observado (modelo de referencia: 366 fechas equiprobables)',value=False)
    if not include_leap:
        valid = valid.loc[~((valid['mes_num']==2)&(valid['dia']==29))]
    n = len(valid)
    freq = valid['fecha_cumple'].value_counts()
    matches = freq[freq>=2]
    groups = len(matches)
    participants = int(matches.sum())
    probability = birthday_probability(n,366 if include_leap else 365)
    a,b,c,d = st.columns(4)
    a.metric('Respuestas recibidas',len(df))
    b.metric('Cumpleaños válidos analizados',n)
    c.metric('Probabilidad teórica',f'{probability:.2%}')
    d.metric('Coincidencia observada','Sí' if groups else 'No')
    st.caption('Modelo de referencia: cumpleaños independientes y fechas equiprobables. En la opción de 365 días se excluyen del análisis observado los nacimientos del 29 de febrero.')
    st.write(f'**Resultado:** {groups} fecha(s) compartida(s), con {participants} participante(s) involucrado(s) en coincidencias. Probabilidad teórica para {n} cumpleaños analizados: **{probability:.2%}**.')
    if groups:
        st.dataframe(pd.DataFrame({'Día/mes':matches.index,'Número de respuestas':matches.values}),hide_index=True,use_container_width=True)
    else:
        st.info('No se observó ninguna coincidencia en este conjunto; ello es posible incluso cuando la probabilidad teórica es alta.')
    if n:
        by_month = valid['mes'].value_counts().reindex(MONTH_NAMES,fill_value=0).rename_axis('Mes').reset_index(name='Frecuencia')
        st.plotly_chart(px.bar(by_month,x='Mes',y='Frecuencia',title='Frecuencia de cumpleaños por mes'),use_container_width=True)
    st.subheader('2. Simulación de la paradoja')
    runs = st.slider('Número de grupos simulados',100,10000,2000,100,key='mot_runs')
    seed = st.number_input('Semilla aleatoria reproducible',min_value=0,max_value=999999,value=42,key='mot_seed')
    if n>=2:
        rng=np.random.default_rng(int(seed))
        sim=rng.integers(0,366 if include_leap else 365,size=(runs,n))
        coincidence=np.array([len(np.unique(row))<n for row in sim])
        empirical_rate=float(coincidence.mean())
        st.metric('Frecuencia de coincidencia en grupos simulados',f'{empirical_rate:.2%}')
        st.caption(f'La frecuencia simulada ({empirical_rate:.2%}) se aproxima a la probabilidad teórica ({probability:.2%}) al repetir el experimento. No debe confundirse con la única observación del salón.')
    st.subheader('3. Distribuciones de frecuencias y estadística descriptiva')
    selected = st.selectbox('Variable numérica',list(LABELS),format_func=lambda x: LABELS[x],key='mot_numeric')
    values = df[selected].dropna()
    if values.empty:
        st.warning('No hay respuestas numéricas válidas para esta variable.')
    else:
        bins = st.slider('Número de intervalos',3,15,6,key='mot_bins')
        edges=np.linspace(float(values.min()),float(values.max()),bins+1) if values.max()!=values.min() else np.array([values.min()-.5,values.max()+.5])
        cats=pd.cut(values,bins=edges,include_lowest=True,duplicates='drop')
        counts=cats.value_counts(sort=False)
        table=pd.DataFrame({'Intervalo':counts.index.astype(str),'f':counts.values})
        table['fr']=table['f']/len(values)
        table['%']=(table['fr']*100).round(2)
        table['F acumulada']=table['f'].cumsum()
        st.dataframe(table,hide_index=True,use_container_width=True)
        st.plotly_chart(px.histogram(x=values,nbins=bins,labels={'x':LABELS[selected],'y':'Frecuencia'},title=f'Distribución: {LABELS[selected]}'),use_container_width=True)
        stats=pd.DataFrame({'Medida':['n válido','Media','Mediana','Moda(s)','Mínimo','Máximo','Rango','Desviación estándar muestral','Varianza muestral','Q1','Q3'], 'Valor':[len(values),round(values.mean(),3),round(values.median(),3),', '.join(map(str,values.mode().head(5).tolist())),values.min(),values.max(),round(values.max()-values.min(),3),round(values.std(ddof=1),3) if len(values)>1 else None,round(values.var(ddof=1),3) if len(values)>1 else None,round(values.quantile(.25),3),round(values.quantile(.75),3)]})
        st.dataframe(stats,hide_index=True,use_container_width=True)
    st.subheader('4. Probabilidad empírica y condicional')
    valid_grades=df['promedio'].notna()
    event=df['promedio']>=8
    count=int((event&valid_grades).sum()); total=int(valid_grades.sum())
    st.write(f'**P(promedio ≥ 8):** {count}/{total} = {count/total:.2%}' if total else 'No hay promedios válidos para calcular probabilidades.')
    condition=df['estudio']>=10
    subset=df['estudio'].notna() & condition & valid_grades
    den=int(subset.sum());num=int((subset&event).sum())
    st.write(f'**P(promedio ≥ 8 | estudio ≥ 10 h/semana):** {num}/{den} = {num/den:.2%}' if den else 'No hay registros válidos en el subgrupo de estudio ≥ 10 horas.')
    st.caption('Son proporciones descriptivas de quienes respondieron; no implican representatividad estadística ni relación causal.')
    st.subheader('5. Exploración de asociaciones educativas')
    pair=df[['estudio','promedio']].dropna()
    if len(pair)>=3:
        st.plotly_chart(px.scatter(pair,x='estudio',y='promedio',labels={'estudio':'Horas de estudio independiente por semana','promedio':'Promedio académico'},title='Horas de estudio y promedio académico (autodeclarado)'),use_container_width=True)
        st.caption(f'Correlación de Pearson (exploratoria): r = {pair.corr().iloc[0,1]:.3f}; n = {len(pair)}. No permite establecer causalidad.')
    else:
        st.info('Se requieren al menos tres pares válidos de horas de estudio y promedio.')
    st.subheader('6. Reflexión pedagógica')
    st.markdown('**Preguntas para discutir:** ¿por qué una coincidencia observada no demuestra una probabilidad? ¿Qué cambia al repetir el experimento? ¿Qué sesgos tienen los promedios autodeclarados? ¿Qué decisiones educativas NO sería válido tomar con esta encuesta?')
    st.caption('Privacidad: no se muestran filas individuales ni se almacenan archivos en una base de datos del módulo. Evita compartir públicamente resultados de subgrupos muy pequeños.')
    output=df.drop(columns=['mes_num','cumple_valido','fecha_cumple']).to_csv(index=False).encode('utf-8-sig')
    st.download_button('Descargar datos normalizados (CSV)',output,'motivacion_datos_normalizados.csv','text/csv')
