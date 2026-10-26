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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
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

def frequency_table(values, bins):
    """Intervalos cerrados a la izquierda y abiertos a la derecha, salvo el último."""
    vals = values.to_numpy(dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(vals, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    # No se utilizan los intervalos textuales de pandas, que introducen
    # artefactos binarios como 6.089999999999999.
    intervals = [f"[{a:.2f}, {b:.2f}{']' if i == len(counts)-1 else ')'}" for i,(a,b) in enumerate(zip(edges[:-1],edges[1:]))]
    table = pd.DataFrame({'Intervalo':intervals,'f':counts})
    table['fr'] = np.round(counts / len(vals), 4)
    table['%'] = np.round(counts / len(vals) * 100, 2)
    table['F acumulada'] = np.cumsum(counts)
    table['Fr acumulada'] = np.round(np.cumsum(counts) / len(vals),4)
    return edges, mids, counts, table

def frequency_figure(edges, mids, counts, label):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mids, y=counts, width=np.diff(edges),
                         name='Histograma', marker_color='#2A72BA', opacity=0.67,
                         customdata=np.column_stack([edges[:-1],edges[1:]]),
                         hovertemplate='Intervalo: %{customdata[0]:.2f} a %{customdata[1]:.2f}<br>Frecuencia: %{y}<extra></extra>'))
    fig.add_trace(go.Scatter(x=mids,y=counts,mode='lines+markers',name='Polígono de frecuencias',
                             line=dict(color='#E4572E',width=3),marker=dict(size=9),
                             hovertemplate='Marca de clase: %{x:.2f}<br>Frecuencia: %{y}<extra></extra>'))
    fig.update_layout(title=f'Histograma y polígono de frecuencias: {label}',
                      xaxis_title=label,yaxis_title='Frecuencia',barmode='overlay',
                      hovermode='x unified',legend=dict(orientation='h',y=1.12),
                      margin=dict(t=95,b=55),dragmode='zoom')
    fig.update_xaxes(range=[edges[0]-(edges[-1]-edges[0])*.03,edges[-1]+(edges[-1]-edges[0])*.03])
    fig.update_yaxes(rangemode='tozero',dtick=1 if max(counts)<=20 else None)
    return fig

def birthday_explanation(n, days):
    return (f"La paradoja del cumpleaños establece que, en un grupo de {n} personas, "
            "la probabilidad de que al menos dos compartan el mismo día y mes de nacimiento "
            "puede ser sorprendentemente elevada. Para calcularla se utiliza el evento complementario: "
            "que todos tengan fechas distintas. Bajo el supuesto de cumpleaños independientes "
            f"y {days} fechas equiprobables, se multiplican las probabilidades de no coincidencia "
            "para cada nueva persona y se resta el resultado de uno. Esta probabilidad corresponde "
            "al modelo teórico, no a la proporción de personas con cumpleaños repetidos ni "
            "a la frecuencia observada en un solo grupo.")


def birthday_illustration(n, days):
    """Figura original generada con Matplotlib, sin imágenes externas."""
    sizes = list(range(2, min(days, max(60, n)) + 1))
    probs = [birthday_probability(k, days) * 100 for k in sizes]
    fig, ax = plt.subplots(figsize=(7.5, 2.65))
    ax.plot(sizes, probs, color='#155A9A', linewidth=2.5)
    ax.axhline(50, color='#888888', linestyle='--', linewidth=1, label='Umbral del 50 %')
    if n >= 2 and n <= days:
        ax.scatter([n], [birthday_probability(n, days)*100], color='#E4572E', s=75, zorder=5)
        ax.annotate(f'n={n}: {birthday_probability(n, days):.2%}', (n, birthday_probability(n, days)*100),
                    xytext=(-90,-28),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Número de personas', ylabel='Probabilidad (%)', title='Cómo crece la probabilidad de coincidencia')
    ax.set_ylim(0, 105);ax.set_xlim(2,max(sizes));ax.grid(alpha=.18)
    fig.tight_layout()
    return fig


def academic_birthday_formula(days):
    """Fórmula matemática vectorial rasterizada a alta resolución para ReportLab."""
    fig, ax = plt.subplots(figsize=(10.2, 1.35))
    ax.axis('off')
    equation = (r'$P(\mathrm{coincidencia}) = 1 - '
                r'\prod_{k=0}^{n-1} \frac{D-k}{D}$')
    ax.text(0.5, 0.53, equation, ha='center', va='center', fontsize=24, color='#183153')
    fig.patch.set_facecolor('white')
    result = io.BytesIO()
    fig.savefig(result, format='png', dpi=220, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    result.seek(0)
    return result


def make_motivation_pdf(df, valid, matches, probability, empirical_rate, table, edges, mids, counts, selected, sim_runs, seed, days=365, synthetic=False):
    """Informe autónomo sin Kaleido ni bibliotecas del sistema."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf,pagesize=(21*cm,29.7*cm),leftMargin=1.8*cm,rightMargin=1.8*cm,
                            topMargin=1.8*cm,bottomMargin=1.8*cm)
    sty=getSampleStyleSheet()
    story=[]
    def para(t,style='Normal'):
        story.append(Paragraph(str(t),sty[style]));story.append(Spacer(1,0.22*cm))
    logos=[]
    root=__import__('pathlib').Path(__file__).parent
    for fn in ['unam.png','cuaed.png','linea_investigacion_pdf.png']:
        path=root/'assets'/fn
        if path.exists():
            from PIL import Image as PILImage
            with PILImage.open(path) as im: w,h=im.size
            target_w=3.0*cm
            logos.append(Image(str(path),width=target_w,height=target_w*h/w))
    if logos:
        row=Table([logos],colWidths=[5.6*cm]*len(logos),hAlign='CENTER')
        story.append(row);story.append(Spacer(1,0.5*cm))
    para('LA-Lab v0.6.17 | Motivación', 'Title')
    para('Encuesta, paradoja del cumpleaños y estadística descriptiva', 'Heading2')
    para(('Reporte exploratorio con datos sintéticos: demostración, no estudiantes reales.' if synthetic else 'Reporte exploratorio con respuestas cargadas por el usuario.') + ' No permite inferir causalidad ni fundamentar decisiones individuales.')
    para('1. Paradoja del cumpleaños','Heading2')
    para(f'Respuestas: {len(df)}. Cumpleaños válidos analizados: {len(valid)}. Fechas compartidas: {len(matches)}. Personas involucradas: {int(matches.sum())}.')
    para(f'Probabilidad teórica: {probability:.2%}. Simulación: {empirical_rate:.2%} en {sim_runs} grupos (semilla {seed}).')
    para(birthday_explanation(len(valid),days))
    para('Modelo matemático de la probabilidad de coincidencia:', 'Heading3')
    formula_png = academic_birthday_formula(days)
    story.append(Image(formula_png, width=14.8*cm, height=1.95*cm, hAlign='CENTER'))
    story.append(Spacer(1, .12*cm))
    para('Donde <b>n</b> es el número de personas y <b>D</b> es el número de fechas posibles '
         f'(<b>D = {days}</b> en este análisis). La expresión es válida para 0 ≤ n ≤ D; '
         'si n &gt; D, la coincidencia es segura. Se supone independencia y equiprobabilidad de los cumpleaños.')
    fig_birth=birthday_illustration(len(valid),days)
    birth_png=io.BytesIO();fig_birth.savefig(birth_png,format='png',dpi=145);plt.close(fig_birth);birth_png.seek(0)
    story.append(Image(birth_png,width=15.8*cm,height=5.6*cm));story.append(Spacer(1,.3*cm))
    para('La observación de una coincidencia en un solo grupo no demuestra la probabilidad teórica. La simulación repite el experimento con grupos independientes.')
    if len(matches):
        t=Table([['Cumpleaños','Respuestas']]+[[str(k),str(v)] for k,v in matches.items()],colWidths=[8*cm,6*cm],repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF1F9')),('GRID',(0,0),(-1,-1),.35,colors.lightgrey),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.append(t);story.append(Spacer(1,.4*cm))
    para('2. Distribución de frecuencias','Heading2')
    para(f'Variable: {LABELS[selected]}. Intervalos: {len(counts)}. Registros válidos: {int(counts.sum())}.')
    data=[['Intervalo','f','fr','%','F acum.','Fr acum.']]+[[str(r['Intervalo']),str(r['f']),f"{r['fr']:.4f}",f"{r['%']:.2f}",str(r['F acumulada']),f"{r['Fr acumulada']:.4f}"] for _,r in table.iterrows()]
    t=Table(data,colWidths=[4.7*cm,1.3*cm,2.1*cm,1.9*cm,2.3*cm,2.5*cm],repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF1F9')),('GRID',(0,0),(-1,-1),.35,colors.lightgrey),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(t);story.append(Spacer(1,.35*cm))
    fig,ax=plt.subplots(figsize=(8,3.5))
    ax.bar(mids,counts,width=np.diff(edges),align='center',alpha=.65,edgecolor='white',label='Histograma')
    ax.plot(mids,counts,'o-',color='#E4572E',linewidth=2,label='Polígono de frecuencias')
    ax.set(xlabel=LABELS[selected],ylabel='Frecuencia');ax.legend();fig.tight_layout()
    png=io.BytesIO();fig.savefig(png,format='png',dpi=145);plt.close(fig);png.seek(0)
    story.append(Image(png,width=16.8*cm,height=7.35*cm));story.append(Spacer(1,.25*cm))
    para('f: frecuencia absoluta; fr: frecuencia relativa; %: porcentaje; F acumulada: número acumulado de observaciones; Fr acumulada: proporción acumulada.')
    vals=df[selected].dropna()
    para(f'Media: {vals.mean():.2f}; mediana: {vals.median():.2f}; desviación estándar muestral: {vals.std(ddof=1):.2f}.' if len(vals)>1 else f'Media: {vals.mean():.2f}.')
    para('3. Estimaciones iniciales de probabilidad','Heading2')
    guesses = df['estimacion'].dropna().astype(str).str.strip()
    if len(guesses):
        guess_counts = guesses.value_counts()
        fig_g,ax_g=plt.subplots(figsize=(8,2.9))
        ax_g.bar(range(len(guess_counts)),guess_counts.values,color='#587DC1')
        ax_g.set_xticks(range(len(guess_counts)),guess_counts.index,rotation=20,ha='right',fontsize=8)
        ax_g.set_ylabel('Respuestas');ax_g.set_title('Estimación previa de los participantes')
        fig_g.tight_layout();gpng=io.BytesIO();fig_g.savefig(gpng,format='png',dpi=145);plt.close(fig_g);gpng.seek(0)
        story.append(Image(gpng,width=15.8*cm,height=5.7*cm));story.append(Spacer(1,.2*cm))
        para(f'Comparación: la probabilidad teórica para {len(valid)} personas es {probability:.2%}. Las estimaciones son intervalos, no valores puntuales.')
    else:
        para('No se registraron estimaciones iniciales de probabilidad.')
    para('4. Probabilidad empírica y analítica educativa','Heading2')
    grades=df['promedio'].dropna()
    if len(grades):
        para(f'Proporción con promedio igual o superior a 8: {(grades>=8).sum()}/{len(grades)} = {(grades>=8).mean():.2%}.')
    pair=df[['estudio','promedio']].dropna()
    if len(pair)>=3 and pair['estudio'].nunique()>1 and pair['promedio'].nunique()>1:
        para(f'Correlación exploratoria entre horas de estudio y promedio: r = {pair.corr().iloc[0,1]:.3f}, n = {len(pair)}. No demuestra causalidad.')
    para('5. Conclusión automática y reflexión pedagógica','Heading2')
    para(f'En el conjunto analizado se identificaron {len(matches)} fechas compartidas entre {len(valid)} cumpleaños válidos. '
         f'La probabilidad teórica de al menos una coincidencia es {probability:.2%}; '
         f'la frecuencia simulada con {sim_runs} grupos fue {empirical_rate:.2%}. '
         'La comparación ilustra el papel de la repetición experimental. Los hábitos y promedios '
         'autodeclarados describen únicamente a quienes respondieron y no permiten conclusiones causales.')
    para('¿Por qué un evento observado no demuestra una probabilidad? ¿Qué cambia al repetir el experimento? ¿Qué limitaciones tienen los datos autodeclarados? ¿Qué decisiones educativas no deben tomarse con esta encuesta?')
    doc.build(story)
    return buf.getvalue()

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
    days=366 if include_leap else 365
    st.caption('Modelo de referencia: cumpleaños independientes y fechas equiprobables. En la opción de 365 días se excluyen del análisis observado los nacimientos del 29 de febrero.')
    st.markdown('**¿En qué consiste la paradoja del cumpleaños?**')
    st.write(birthday_explanation(n,days))
    st.latex(r'P(\text{al menos una coincidencia})=1-\prod_{k=0}^{n-1}\frac{D-k}{D}')
    st.caption(f'En este conjunto, n = {n} y D = {days}. Para n > D, la coincidencia es segura.')
    st.pyplot(birthday_illustration(n,days),clear_figure=True)
    st.write(f'**Resultado:** {groups} fecha(s) compartida(s), con {participants} participante(s) involucrado(s) en coincidencias. Probabilidad teórica para {n} cumpleaños analizados: **{probability:.2%}**.')
    with st.expander('¿Cómo se calcula la probabilidad teórica?'):
        st.latex(r'P(\text{al menos una coincidencia})=1-\prod_{k=0}^{n-1}\frac{D-k}{D}')
        st.write(f'En esta actividad: **n = {n}** personas y **D = {366 if include_leap else 365}** fechas posibles. Primero se calcula la probabilidad de que todos tengan cumpleaños diferentes y después se resta de 1.')
        st.caption('Supuestos: independencia y equiprobabilidad de fechas. Las fechas reales no son exactamente equiprobables.')
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
    st.subheader('Estimaciones iniciales del grupo')
    guesses = df['estimacion'].dropna().astype(str).str.strip()
    if len(guesses):
        guesses_freq = guesses.value_counts().rename_axis('Estimación').reset_index(name='Respuestas')
        fig_guess = px.bar(guesses_freq,x='Estimación',y='Respuestas',text='Respuestas',title='Probabilidad estimada antes de conocer la respuesta')
        fig_guess.update_layout(xaxis_title='Intervalo de probabilidad estimado',yaxis_title='Número de respuestas')
        st.plotly_chart(fig_guess,use_container_width=True)
        st.caption(f'La probabilidad teórica es {probability:.2%}. Las estimaciones son categorías de intervalo, por lo que no se les asigna artificialmente un valor puntual.')
    else:
        st.info('No hay estimaciones iniciales registradas.')
    st.subheader('3. Distribuciones de frecuencias y estadística descriptiva')
    selected = st.selectbox('Variable numérica',list(LABELS),format_func=lambda x: LABELS[x],key='mot_numeric')
    values = df[selected].dropna()
    if values.empty:
        st.warning('No hay respuestas numéricas válidas para esta variable.')
    else:
        bins = st.slider('Número de intervalos',3,15,6,key='mot_bins')
        edges,mids,counts,table = frequency_table(values,bins)
        st.dataframe(table,hide_index=True,use_container_width=True)
        st.caption('f = frecuencia absoluta; fr = frecuencia relativa; % = porcentaje; F acumulada = número de observaciones hasta ese intervalo; Fr acumulada = proporción acumulada. La suma de f debe ser igual al número de datos válidos.')
        st.plotly_chart(frequency_figure(edges,mids,counts,LABELS[selected]),use_container_width=True)
        st.caption('Interactúa con la figura: pasa el cursor para ver marcas de clase y frecuencias; usa la leyenda para ocultar el histograma o el polígono y las herramientas para ampliar o descargar la imagen. Las clases son cerradas a la izquierda y abiertas a la derecha, excepto la última.')
        stats=pd.DataFrame({'Medida':['n válido','Media','Mediana','Moda(s)','Mínimo','Máximo','Rango','Desviación estándar muestral','Varianza muestral','Q1','Q3'], 'Valor':[len(values),round(values.mean(),3),round(values.median(),3),', '.join(map(str,values.mode().head(5).tolist())),values.min(),values.max(),round(values.max()-values.min(),3),round(values.std(ddof=1),3) if len(values)>1 else None,round(values.var(ddof=1),3) if len(values)>1 else None,round(values.quantile(.25),3),round(values.quantile(.75),3)]})
        st.dataframe(stats,hide_index=True,use_container_width=True)
    if not values.empty and n>=2:
        try:
            pdf=make_motivation_pdf(df,valid,matches,probability,empirical_rate,table,edges,mids,counts,selected,runs,int(seed),days,source.startswith('Ejemplo'))
            st.download_button('Descargar informe de Motivación (PDF)',pdf,'LA-Lab_Motivacion_v0617.pdf','application/pdf',key='mot_pdf')
        except Exception as exc:
            st.warning(f'No se pudo generar el informe PDF: {exc}')
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
    st.subheader('6. Conclusión y reflexión pedagógica')
    st.write(f'**Síntesis automática:** de {n} cumpleaños válidos, se identificaron {groups} fechas compartidas '
             f'que involucran a {participants} personas. La probabilidad teórica de que exista al menos '
             f'una coincidencia es {probability:.2%}; en {runs} grupos simulados se observó '
             f'una frecuencia de {empirical_rate:.2%}. Estos resultados ilustran el contraste entre '
             'modelo probabilístico, simulación y observación única. Las asociaciones educativas '
             'exploradas no implican causalidad.')
    st.markdown('**Preguntas para discutir:** ¿por qué una coincidencia observada no demuestra una probabilidad? ¿Qué cambia al repetir el experimento? ¿Qué sesgos tienen los promedios autodeclarados? ¿Qué decisiones educativas NO sería válido tomar con esta encuesta?')
    st.caption('Privacidad: no se muestran filas individuales ni se almacenan archivos en una base de datos del módulo. Evita compartir públicamente resultados de subgrupos muy pequeños.')
    output=df.drop(columns=['mes_num','cumple_valido','fecha_cumple']).to_csv(index=False).encode('utf-8-sig')
    st.download_button('Descargar datos normalizados (CSV)',output,'motivacion_datos_normalizados.csv','text/csv')
