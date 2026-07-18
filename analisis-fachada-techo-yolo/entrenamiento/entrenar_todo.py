"""
Entrena las 7 categorías de modelos una por una (no en paralelo), cada una usando
las 3 GPUs disponibles. Pensado para dejarlo corriendo sin supervisión — no pide
confirmación en ningún punto, y si un entrenamiento falla, registra el error y
sigue con el siguiente en vez de detener todo el proceso.

Uso (desde la raíz del proyecto), para que siga corriendo aunque cierres la sesión:
    nohup python3 entrenamiento/entrenar_todo.py > entrenamiento/log_entrenar_todo.txt 2>&1 &

Para continuar la cola sin repetir modelos ya hechos (por ejemplo si entrenaste
'fachada' aparte a mano y solo quieres que siga con lo que falta):
    nohup python3 entrenamiento/entrenar_todo.py --desde ventanas > entrenamiento/log_entrenar_todo.txt 2>&1 &

'--desde <id>' salta todo lo que esté antes de ese id en el PLAN (se marca como ya
completado, no se vuelve a correr) y, antes de arrancar el primero de la lista,
espera a que no haya otro entrenar_*.py corriendo todavía (candado en
entrenamiento/.locks/) — así no compite por las mismas 3 GPUs con algo que ya esté
entrenando (por ejemplo, un fine-tuning lanzado a mano aparte de este script).

Progreso: además de lo que cada script ya imprime (redirigido al log de arriba) y
de runs/detect/entrenamiento_<id>/results.csv (que Ultralytics actualiza por época,
igual que antes), este script escribe entrenamiento/estado_entrenamiento.json con
qué modelo está corriendo y cuáles ya terminaron/fallaron. Lo lee
entrenamiento/monitor_entrenamiento.py (interfaz gráfica de progreso), pero es JSON
plano, así que también se puede revisar a mano o desde cualquier otro script.
"""

import argparse
import json
import os
import subprocess
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESTADO_PATH = os.path.join(RAIZ, 'entrenamiento', 'estado_entrenamiento.json')
LOCKS_DIR = os.path.join(RAIZ, 'entrenamiento', '.locks')

# Orden de entrenamiento. 'epocas' es solo informativo para el monitor (el número
# real ya está fijo dentro de cada script) — si cambias uno, cambia también aquí.
PLAN = [
    {'id': 'techo',           'nombre': 'Techos (satelital)',               'script': 'entrenar_techo.py',           'epocas': 200},
    {'id': 'fachada',         'nombre': 'Fachadas (estructura + daños)',    'script': 'entrenar_fachada.py',         'epocas': 500},
    {'id': 'ventanas',        'nombre': 'Ventanas',                         'script': 'entrenar_ventanas.py',        'epocas': 500},
    {'id': 'fachada_general', 'nombre': 'Fachada general (arquitectónico)', 'script': 'entrenar_fachada_general.py', 'epocas': 500},
    {'id': 'danos',           'nombre': 'Daños/deterioro',                  'script': 'entrenar_danos.py',           'epocas': 500},
    {'id': 'senales',         'nombre': 'Señalamiento vial',                'script': 'entrenar_senales.py',         'epocas': 500},
    {'id': 'calles',          'nombre': 'Calles (vehículos/peatones)',      'script': 'entrenar_calles.py',          'epocas': 500},
]


def escribir_estado(estado):
    """Escritura atómica (escribe a un .tmp y renombra) para que el monitor nunca
    lea un JSON a medio escribir mientras este script sigue corriendo."""
    tmp = ESTADO_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(estado, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ESTADO_PATH)


def candados_activos():
    """Nombres de los candados en entrenamiento/.locks/ cuyo proceso dueño (por PID)
    todavía existe — ver el mismo mecanismo en cada entrenar_*.py."""
    if not os.path.isdir(LOCKS_DIR):
        return []
    activos = []
    for archivo in os.listdir(LOCKS_DIR):
        ruta = os.path.join(LOCKS_DIR, archivo)
        try:
            with open(ruta) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            activos.append(archivo)
        except (ValueError, ProcessLookupError, PermissionError, FileNotFoundError):
            pass
    return activos


def esperar_gpus_libres(log_cb):
    """Antes de arrancar el primer modelo de la cola, espera a que no haya otro
    entrenar_*.py corriendo (por ejemplo un fine-tuning lanzado a mano aparte de
    este orquestador) — todos usan las mismas 3 GPUs, correr dos a la vez las
    satura y puede corromper resultados (ya pasó una vez, por eso el candado)."""
    primera_vez = True
    while True:
        activos = candados_activos()
        if not activos:
            return
        if primera_vez:
            log_cb(f"Esperando a que termine lo que ya está corriendo ({', '.join(activos)}) antes de empezar...")
            primera_vez = False
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desde', default=None,
                         help="id del primer modelo a entrenar (techo/fachada/ventanas/fachada_general/danos/senales/calles) "
                              "— salta los anteriores en el PLAN, se marcan como ya completados sin volver a correrlos.")
    args = parser.parse_args()

    ids_plan = [p['id'] for p in PLAN]
    if args.desde:
        if args.desde not in ids_plan:
            print(f"id desconocido: '{args.desde}'. Opciones: {', '.join(ids_plan)}")
            sys.exit(1)
        idx_inicio = ids_plan.index(args.desde)
    else:
        idx_inicio = 0

    saltados = [p['id'] for p in PLAN[:idx_inicio]]
    plan_a_correr = PLAN[idx_inicio:]

    estado = {
        'inicio': time.strftime('%Y-%m-%d %H:%M:%S'),
        'fin': None,
        'actual': None,
        'plan': PLAN,
        'pendientes': [p['id'] for p in plan_a_correr],
        'completados': list(saltados),  # ya hechos antes (a mano o en una corrida previa), no se repiten
        'fallidos': [],
        'log': [],
    }
    if saltados:
        estado['log'].append(f"{time.strftime('%H:%M:%S')} — arrancando desde '{args.desde}', "
                              f"se saltan (ya completados): {', '.join(saltados)}")
    escribir_estado(estado)
    print(f"Entrenamiento encadenado iniciado — {len(plan_a_correr)} modelos por correr, uno por uno, 3 GPUs cada uno.")

    def log_espera(msg):
        estado['log'].append(f"{time.strftime('%H:%M:%S')} — {msg}")
        escribir_estado(estado)
        print(msg)

    esperar_gpus_libres(log_espera)

    for paso in plan_a_correr:
        estado['actual'] = paso['id']
        estado['pendientes'].remove(paso['id'])
        # Sin emoji: este mensaje se muestra tal cual en el Text widget de
        # monitor_entrenamiento.py, y algunos entornos X11/Tk truenan al
        # renderizar glifos de emoji (BadLength en RENDER, visto en pruebas).
        msg = f"{time.strftime('%H:%M:%S')} — iniciando '{paso['id']}' ({paso['epocas']} épocas)"
        estado['log'].append(msg)
        escribir_estado(estado)
        print(msg)

        script_path = os.path.join(RAIZ, 'entrenamiento', paso['script'])
        inicio = time.time()
        proceso = subprocess.run([sys.executable, script_path], cwd=RAIZ)
        duracion_min = (time.time() - inicio) / 60

        if proceso.returncode == 0:
            estado['completados'].append(paso['id'])
            msg = f"{time.strftime('%H:%M:%S')} — OK '{paso['id']}' completado ({duracion_min:.1f} min)"
        else:
            estado['fallidos'].append(paso['id'])
            msg = (f"{time.strftime('%H:%M:%S')} — FALLO '{paso['id']}' "
                   f"(código {proceso.returncode}, {duracion_min:.1f} min) — sigo con el siguiente")

        estado['actual'] = None
        estado['log'].append(msg)
        escribir_estado(estado)
        print(msg)

    estado['fin'] = time.strftime('%Y-%m-%d %H:%M:%S')
    resumen = f"{time.strftime('%H:%M:%S')} — Terminado. {len(estado['completados'])} OK, {len(estado['fallidos'])} fallidos."
    estado['log'].append(resumen)
    escribir_estado(estado)
    print(resumen)


if __name__ == '__main__':
    main()
