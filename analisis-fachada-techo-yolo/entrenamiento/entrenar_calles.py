
import atexit
import shutil
import os
import sys
from ultralytics import YOLO

RUN_NAME = 'entrenamiento_calles'
RUN_DIR = os.path.join('runs', 'detect', RUN_NAME)
PESOS_PREVIOS = os.path.join(RUN_DIR, 'weights', 'best.pt')

# Candado: evita que dos corridas de este mismo modelo se pisen (pasó una vez — dos
# procesos escribiendo a la vez a RUN_DIR corrompieron results.csv y los pesos con
# filas/checkpoints entreverados de ambas corridas). Se basa en el PID, no en la sola
# existencia del archivo: si el proceso dueño del candado ya no existe (se cayó, lo
# mataron con -9, etc.), el candado se considera viejo y no bloquea la corrida nueva.
LOCK_PATH = os.path.join('entrenamiento', '.locks', 'entrenamiento_calles.lock')
os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
if os.path.isfile(LOCK_PATH):
    with open(LOCK_PATH) as f:
        pid_previo = f.read().strip()
    try:
        os.kill(int(pid_previo), 0)
        print(f"❌ Ya hay un entrenamiento de 'entrenamiento_calles' corriendo (PID {pid_previo}). "
              "Cancélalo primero o espera a que termine — no se puede correr dos veces a la vez.")
        sys.exit(1)
    except (ValueError, ProcessLookupError, PermissionError):
        pass  # PID no existe o no es válido: candado viejo de una corrida que ya no está, se ignora
with open(LOCK_PATH, 'w') as f:
    f.write(str(os.getpid()))
atexit.register(lambda: os.path.isfile(LOCK_PATH) and os.remove(LOCK_PATH))

# Si ya hay un entrenamiento previo de este modelo, continúa desde ahí (fine-tuning)
# en vez de partir de cero desde yolo11x.pt — por ejemplo si una corrida anterior se
# cortó antes de las épocas pedidas (EarlyStopping de Ultralytics, patience=100 por
# defecto) o se detuvo por cualquier otro motivo, así no se pierde lo ya aprendido.
# La corrida anterior se respalda (no se borra) por si se quiere comparar antes/después.
if os.path.isfile(PESOS_PREVIOS):
    respaldo = RUN_DIR + '_previo'
    if os.path.isdir(respaldo):
        shutil.rmtree(respaldo)
    shutil.move(RUN_DIR, respaldo)  # mover ANTES de fijar modelo_base: la ruta cambia con el move
    modelo_base = os.path.join(respaldo, 'weights', 'best.pt')
    print(f"🔁 Encontrados pesos previos — continúa el entrenamiento desde ahí (fine-tuning): {modelo_base}")
else:
    print("No hay entrenamiento previo de este modelo, se parte de yolo11x.pt (COCO).")
    modelo_base = 'yolo11x.pt'
    if os.path.isdir(RUN_DIR):
        print(f"🗑️  Eliminando carpeta incompleta/vieja en {RUN_DIR}")
        shutil.rmtree(RUN_DIR)

# Modelo de escena de calle (vehículos, peatones, etc., 12 clases). NO es un
# modelo de fachada — es una categoría independiente (carpeta "Facade/streets").
model = YOLO(modelo_base)

print("🚀 Iniciando entrenamiento de Calles...")
results = model.train(
    data='datasets/dataset_calles.yaml',
    epochs=500,
    imgsz=640,
    batch=16,          # Ajusta según memoria de tus GPUs
    device='0,1,2',    # las 3 GPUs — este script ya no corre en paralelo con otros (ver entrenar_todo.py)
    patience=0,        # sin EarlyStopping — se piden 500 épocas a propósito, que se cumplan
    name=RUN_NAME,
    plots=True
)

print("✅ ¡Entrenamiento de Calles completado!")
