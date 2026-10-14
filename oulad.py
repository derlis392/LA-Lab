"""Preparación trazable de tres prácticas OULAD desde el ZIP original público.
No incluye ni fabrica registros educativos. Procesa studentVle por bloques.
"""
import io
import zipfile
import pandas as pd

KEY = ['code_module', 'code_presentation', 'id_student']
NEEDED = {
    12: ['studentInfo.csv', 'studentVle.csv'],
    13: ['studentInfo.csv', 'studentAssessment.csv', 'assessments.csv'],
    14: ['studentInfo.csv', 'studentRegistration.csv'],
}

def prepare_oulad(upload_bytes, case_number):
    if case_number not in NEEDED:
        raise ValueError('Práctica OULAD desconocida.')
    try:
        archive = zipfile.ZipFile(io.BytesIO(upload_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError('El archivo no es un ZIP válido de OULAD.') from exc
    with archive as z:
        names = {name.rsplit('/', 1)[-1].lower(): name for name in z.namelist() if not name.endswith('/')}
        missing = [n for n in NEEDED[case_number] if n.lower() not in names]
        if missing:
            raise ValueError('Faltan tablas oficiales en el ZIP: ' + ', '.join(missing))
        def read(name, **kwargs):
            with z.open(names[name.lower()]) as stream:
                return pd.read_csv(stream, **kwargs)
        info = read('studentInfo.csv')
        for col in KEY + ['final_result']:
            if col not in info.columns:
                raise ValueError('studentInfo.csv no contiene la columna ' + col)
        info = info.drop_duplicates(subset=KEY).copy()
        if case_number == 12:
            groups = []
            with z.open(names['studentvle.csv']) as stream:
                for chunk in pd.read_csv(stream, usecols=KEY+['sum_click'], chunksize=200_000):
                    chunk['sum_click'] = pd.to_numeric(chunk['sum_click'], errors='coerce').fillna(0)
                    groups.append(chunk.groupby(KEY, as_index=False)['sum_click'].sum())
            if not groups:
                raise ValueError('studentVle.csv no tiene registros.')
            activity = pd.concat(groups, ignore_index=True).groupby(KEY, as_index=False)['sum_click'].sum()
            activity = activity.rename(columns={'sum_click':'clics_vle_totales'})
            result = info.merge(activity, on=KEY, how='left', validate='one_to_one')
            result['clics_vle_totales'] = result['clics_vle_totales'].fillna(0)
        elif case_number == 13:
            sa = read('studentAssessment.csv')
            ass = read('assessments.csv')
            needed_sa = ['id_assessment', 'id_student', 'score']
            needed_ass = ['id_assessment', 'code_module', 'code_presentation']
            if any(c not in sa for c in needed_sa) or any(c not in ass for c in needed_ass):
                raise ValueError('Las tablas de evaluación no contienen las claves y puntuaciones necesarias.')
            # No usar final_result como predictor de los resultados de evaluación.
            joined = sa.merge(ass[needed_ass].drop_duplicates('id_assessment'), on='id_assessment', how='inner', validate='many_to_one')
            joined['score'] = pd.to_numeric(joined['score'], errors='coerce')
            agg = joined.groupby(KEY, as_index=False).agg(evaluaciones_entregadas=('score','count'), puntuacion_media=('score','mean'))
            result = info.merge(agg, on=KEY, how='left', validate='one_to_one')
            result['evaluaciones_entregadas'] = result['evaluaciones_entregadas'].fillna(0).astype(int)
        else:
            registration = read('studentRegistration.csv')
            if any(c not in registration for c in KEY + ['date_registration', 'date_unregistration']):
                raise ValueError('studentRegistration.csv no tiene las columnas esperadas.')
            registration = registration[KEY+['date_registration','date_unregistration']].drop_duplicates(KEY)
            result = info.merge(registration, on=KEY, how='left', validate='one_to_one')
            result['registro_retirado'] = result['date_unregistration'].notna().astype(int)
        if result.empty:
            raise ValueError('La combinación de tablas no produjo registros.')
        return result
