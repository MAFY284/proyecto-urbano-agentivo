"""
Interfaz gráfica simple (Tkinter, sin dependencias extra) para ver el progreso del
entrenamiento encadenado lanzado con entrenar_todo.py: qué modelo está entrenando
ahora, en qué época/lote va, porcentaje, tiempo restante estimado, y cuáles ya
terminaron o fallaron.

No necesita que entrenar_todo.py esté corriendo en la misma terminal — solo lee:
  - entrenamiento/estado_entrenamiento.json    (qué modelo está activo, completados, fallidos)
  - runs/detect/entrenamiento_<id>/results.csv (época ya completada y métricas, las escribe Ultralytics)
  - entrenamiento/log_entrenar_todo.txt        (progreso EN VIVO dentro de la época actual — la barra
                                                 de progreso por lote que imprime Ultralytics/tqdm)
El último archivo asume que lanzaste entrenar_todo.py redirigiendo su salida ahí, tal como
recomienda el README:
    nohup python3 entrenamiento/entrenar_todo.py > entrenamiento/log_entrenar_todo.txt 2>&1 &
Si no existe, el monitor sigue funcionando con lo que hay en estado_entrenamiento.json y
results.csv (progreso solo por época completa, sin barra en vivo).

Uso (en otra terminal, mientras entrenar_todo.py corre en background):
    python3 entrenamiento/monitor_entrenamiento.py
"""

import csv
import json
import os
import re
import tkinter as tk
from tkinter import ttk

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESTADO_PATH = os.path.join(RAIZ, 'entrenamiento', 'estado_entrenamiento.json')
LOG_PATH = os.path.join(RAIZ, 'entrenamiento', 'log_entrenar_todo.txt')
INTERVALO_MS = 3000
TAIL_BYTES = 200_000  # suficiente para encontrar la última línea de progreso sin leer el log completo

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
# Ejemplo de línea real (tqdm de Ultralytics):
#   "      1/200       6.9G      1.706      1.521       1.52        331        640: 27% |███| 409/1498 1.1it/s 6:13<16:29"
PROGRESO_RE = re.compile(
    r'^\s*(\d+)/(\d+)\s+[\d.]+G\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+:\s+(\d+)%.*?'
    r'(\d+)/(\d+)\s+[\d.]+\s*(?:it/s|s/it).*?([\d:]+)<([\d:]+)'
)


def leer_estado():
    if not os.path.isfile(ESTADO_PATH):
        return None
    try:
        with open(ESTADO_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # se leyó a medio escribir o aún no existe; se reintenta en el próximo tick


def leer_ultima_epoca(model_id):
    """Última fila de results.csv para el modelo dado (época YA completada). None si aún no hay."""
    csv_path = os.path.join(RAIZ, 'runs', 'detect', f'entrenamiento_{model_id}', 'results.csv')
    if not os.path.isfile(csv_path):
        return None
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            filas = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None
    return filas[-1] if filas else None


def leer_progreso_batch():
    """Última línea de progreso por lote (dentro de la época en curso) en el log
    de salida de entrenar_todo.py. Devuelve None si no hay log o no hay coincidencia
    reciente (p. ej. si el modelo actual está en validación, no entrenamiento)."""
    if not os.path.isfile(LOG_PATH):
        return None
    try:
        with open(LOG_PATH, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            data = f.read().decode('utf-8', errors='ignore')
    except OSError:
        return None

    data = ANSI_RE.sub('', data)
    lineas = re.split(r'[\r\n]', data)
    for linea in reversed(lineas):
        m = PROGRESO_RE.search(linea)
        if m:
            epoca, epoca_tot, pct, batch, batch_tot, transcurrido, restante = m.groups()
            return {
                'epoca': int(epoca), 'epoca_tot': int(epoca_tot), 'pct_epoca': int(pct),
                'batch': int(batch), 'batch_tot': int(batch_tot),
                'transcurrido': transcurrido, 'restante': restante,
            }
    return None


class MonitorEntrenamiento:
    def __init__(self, root):
        self.root = root
        root.title("Monitor de entrenamiento — 7 modelos")
        root.geometry("900x600")

        header = ttk.Frame(root, padding=10)
        header.pack(fill='x')
        self.label_resumen = ttk.Label(header, text="Esperando entrenamiento_estado.json...", font=('', 11, 'bold'))
        self.label_resumen.pack(anchor='w')

        # ── Progreso del modelo que está entrenando ahora ──
        activo = ttk.LabelFrame(root, text="Modelo actual", padding=10)
        activo.pack(fill='x', padx=10, pady=(0, 10))

        self.label_activo = ttk.Label(activo, text="—", font=('', 10, 'bold'))
        self.label_activo.pack(anchor='w')

        ttk.Label(activo, text="Progreso general (todas las épocas):").pack(anchor='w', pady=(8, 0))
        self.barra_general = ttk.Progressbar(activo, orient='horizontal', mode='determinate', maximum=100)
        self.barra_general.pack(fill='x', pady=(2, 0))
        self.label_general = ttk.Label(activo, text="")
        self.label_general.pack(anchor='w')

        ttk.Label(activo, text="Progreso dentro de la época actual:").pack(anchor='w', pady=(8, 0))
        self.barra_epoca = ttk.Progressbar(activo, orient='horizontal', mode='determinate', maximum=100)
        self.barra_epoca.pack(fill='x', pady=(2, 0))
        self.label_epoca = ttk.Label(activo, text="")
        self.label_epoca.pack(anchor='w')

        # ── Tabla con los 7 modelos ──
        cols = ('modelo', 'estado', 'epoca', 'mAP50', 'mAP50-95', 'box_loss')
        self.tabla = ttk.Treeview(root, columns=cols, show='headings', height=8)
        titulos = {
            'modelo': 'Modelo', 'estado': 'Estado', 'epoca': 'Época (completa)',
            'mAP50': 'mAP50', 'mAP50-95': 'mAP50-95', 'box_loss': 'box_loss (train)',
        }
        anchos = {'modelo': 220, 'estado': 110, 'epoca': 130, 'mAP50': 90, 'mAP50-95': 90, 'box_loss': 110}
        for c in cols:
            self.tabla.heading(c, text=titulos[c])
            self.tabla.column(c, width=anchos[c], anchor='center' if c != 'modelo' else 'w')
        self.tabla.pack(fill='x', padx=10, pady=(0, 10))

        ttk.Label(root, text="Registro:", padding=(10, 0)).pack(anchor='w')
        log_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        log_frame.pack(fill='both', expand=True)
        self.log_text = tk.Text(log_frame, height=8, state='disabled', wrap='word', font=('Courier', 9))
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.filas_por_id = {}
        self.actualizar()

    def actualizar(self):
        estado = leer_estado()
        if estado is None:
            self.label_resumen.config(text="Esperando a que arranque entrenar_todo.py...")
            self.root.after(INTERVALO_MS, self.actualizar)
            return

        completados = set(estado.get('completados', []))
        fallidos = set(estado.get('fallidos', []))
        actual = estado.get('actual')

        n_total = len(estado.get('plan', []))
        resumen = f"{len(completados)}/{n_total} completados"
        if fallidos:
            resumen += f" — {len(fallidos)} fallidos"
        if actual:
            resumen += f" — entrenando ahora: {actual}"
        elif estado.get('fin'):
            resumen += f" — terminado {estado['fin']}"
        self.label_resumen.config(text=resumen)

        # ── Panel "modelo actual" ──
        self._actualizar_panel_activo(estado, actual)

        # ── Tabla ──
        if not self.filas_por_id:
            for paso in estado.get('plan', []):
                iid = self.tabla.insert('', 'end', values=(paso['nombre'], '', '', '', '', ''))
                self.filas_por_id[paso['id']] = iid

        for paso in estado.get('plan', []):
            pid = paso['id']
            iid = self.filas_por_id.get(pid)
            if iid is None:
                continue

            if pid in completados:
                estado_txt = 'Listo'
            elif pid in fallidos:
                estado_txt = 'Fallo'
            elif pid == actual:
                estado_txt = 'Entrenando'
            else:
                estado_txt = 'Pendiente'

            fila = leer_ultima_epoca(pid) if pid == actual or pid in completados else None
            if fila:
                epoca_txt = f"{fila.get('epoch', '?')}/{paso['epocas']}"
                mAP50 = fila.get('metrics/mAP50(B)', '')
                mAP5095 = fila.get('metrics/mAP50-95(B)', '')
                box_loss = fila.get('train/box_loss', '')
                mAP50 = f"{float(mAP50):.3f}" if mAP50 else ''
                mAP5095 = f"{float(mAP5095):.3f}" if mAP5095 else ''
                box_loss = f"{float(box_loss):.3f}" if box_loss else ''
            else:
                epoca_txt = mAP50 = mAP5095 = box_loss = ''

            self.tabla.item(iid, values=(paso['nombre'], estado_txt, epoca_txt, mAP50, mAP5095, box_loss))

        log_lineas = estado.get('log', [])
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.insert('end', '\n'.join(log_lineas[-200:]))
        self.log_text.see('end')
        self.log_text.config(state='disabled')

        self.root.after(INTERVALO_MS, self.actualizar)

    def _actualizar_panel_activo(self, estado, actual):
        if not actual:
            self.label_activo.config(text="Ningún modelo entrenando ahora mismo.")
            self.barra_general['value'] = 0
            self.barra_epoca['value'] = 0
            self.label_general.config(text="")
            self.label_epoca.config(text="")
            return

        paso = next((p for p in estado.get('plan', []) if p['id'] == actual), None)
        nombre = paso['nombre'] if paso else actual
        epocas_totales = paso['epocas'] if paso else None

        progreso = leer_progreso_batch()
        if progreso and (not epocas_totales or progreso['epoca_tot'] == epocas_totales):
            self.label_activo.config(text=f"{nombre} — época {progreso['epoca']}/{progreso['epoca_tot']}")

            pct_epoca = progreso['pct_epoca']
            self.barra_epoca['value'] = pct_epoca
            self.label_epoca.config(
                text=f"Lote {progreso['batch']}/{progreso['batch_tot']} — {pct_epoca}% — "
                     f"transcurrido {progreso['transcurrido']}, restante ≈ {progreso['restante']}"
            )

            epoca_tot = progreso['epoca_tot']
            pct_general = ((progreso['epoca'] - 1) + pct_epoca / 100) / epoca_tot * 100
            self.barra_general['value'] = max(0, min(100, pct_general))
            self.label_general.config(text=f"Época {progreso['epoca']}/{epoca_tot} ({pct_general:.1f}% del total)")
        else:
            # Sin línea de progreso reciente (p. ej. validando al final de la época, o log no
            # disponible): usa la última época YA completada de results.csv como referencia.
            fila = leer_ultima_epoca(actual)
            if fila and epocas_totales:
                epoca_actual = int(float(fila.get('epoch', 0)))
                pct_general = epoca_actual / epocas_totales * 100
                self.label_activo.config(text=f"{nombre} — época {epoca_actual}/{epocas_totales} (validando / entre épocas)")
                self.barra_general['value'] = max(0, min(100, pct_general))
                self.label_general.config(text=f"Época {epoca_actual}/{epocas_totales} ({pct_general:.1f}% del total)")
            else:
                self.label_activo.config(text=f"{nombre} — preparando (todavía sin época completa)")
                self.barra_general['value'] = 0
                self.label_general.config(text="")
            self.barra_epoca['value'] = 0
            self.label_epoca.config(text="Esperando datos de progreso por lote...")


if __name__ == '__main__':
    root = tk.Tk()
    MonitorEntrenamiento(root)
    root.mainloop()
