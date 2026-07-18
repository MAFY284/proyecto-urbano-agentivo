# Módulo: Entrenamiento de modelos YOLO11 — Fachadas, Techos y Daños

Módulo de origen del pool de detección del sistema (autor: Mario). Contiene el **pipeline
completo de organización de datasets y entrenamiento** de los 7 modelos YOLO11 que usa la
plataforma, junto con la evidencia de cada corrida (métricas, curvas y matrices de
confusión en `runs/detect/`).

> **Los pesos entrenados no viven aquí** — están en el almacén único del proyecto,
> [`config/checkpoints/`](../../config/checkpoints/) (raíz del repositorio, vía Git LFS).
> La inferencia, el servidor web y el score de riesgo también viven en la raíz
> (`servidor.py`, `src/`); este módulo se conserva para reentrenar o ampliar los modelos.

## Contenido

```
├── organizar_datasets.py         # Normaliza datasets_fuente/ a formato YOLO-bbox
├── descargar_sentinel.py         # Descarga imágenes satelitales del área de estudio
├── CMP_facade_DB_base/           # Fachadas con anotaciones XML propias (378 imágenes)
├── entrenamiento/
│   ├── entrenar_techo.py         # Techos — 200 épocas
│   ├── entrenar_fachada.py       # Fachadas (fusión) — 500 épocas
│   ├── entrenar_ventanas.py      # Ventanas — 500 épocas
│   ├── entrenar_fachada_general.py  # Elementos arquitectónicos — 500 épocas
│   ├── entrenar_danos.py         # Daños/deterioro — 500 épocas
│   ├── entrenar_senales.py       # Señalamiento vial — 500 épocas
│   ├── entrenar_calles.py        # Escena de calle — 500 épocas
│   ├── entrenar_todo.py          # Orquesta los 7 en secuencia, sin supervisión
│   └── monitor_entrenamiento.py  # Interfaz gráfica (Tkinter) de progreso
└── runs/detect/                  # Evidencia de entrenamiento: métricas por modelo
                                  # (results.csv, curvas PR/F1, matrices de confusión)
```

## Organización de datasets

```bash
python3 organizar_datasets.py --dry-run   # previsualiza conteos sin copiar nada
python3 organizar_datasets.py             # organiza de verdad
```

Normaliza fuentes heterogéneas (YOLO-bbox, YOLO-segmentación, COCO JSON y máscaras raster)
a YOLO-bbox, separadas en 7 categorías: techos, ventanas, fachada general (25 clases),
daños (10 clases), fachadas-fusión (17 clases — la que usa el análisis principal), señales
(47 clases) y calles (12 clases). Es idempotente: puede re-ejecutarse tras agregar fuentes
nuevas sin reprocesar lo existente.

> **Al modificar una lista de clases (`CLASES_*`):** agrega clases nuevas siempre al final.
> Si se inserta una clase en medio, las etiquetas ya generadas quedan apuntando al índice
> equivocado — borra la carpeta de destino afectada y regenera desde cero.

Los datasets crudos (`datasets_fuente/`) y procesados (`datasets/`) **no** están en el
repositorio por su tamaño (decenas de GB); solo hacen falta para reentrenar.

## Entrenamiento

```bash
# Los 7 modelos en secuencia, sin supervisión:
nohup python3 entrenamiento/entrenar_todo.py > entrenamiento/log_entrenar_todo.txt 2>&1 &
python3 entrenamiento/monitor_entrenamiento.py   # opcional: progreso en tiempo real

# Un modelo individual:
python3 entrenamiento/entrenar_techo.py

# Reanudar la cola desde un punto:
python3 entrenamiento/entrenar_todo.py --desde ventanas
```

- Cada script usa todas las GPUs disponibles (`device='0,1,2'`) con `patience=0`
  (sin interrupción anticipada).
- Si ya existe un `best.pt` previo para la categoría, continúa desde esos pesos
  (fine-tuning) y respalda la corrida anterior en `..._previo/`.
- Al terminar, copia el `best.pt` resultante al almacén único para que el sistema lo use:

```bash
cp runs/detect/entrenamiento_techo/weights/best.pt ../../config/checkpoints/techo_best.pt
```

## Dependencias

```bash
pip install -r requirements.txt
```

Desarrollado y probado con 3× NVIDIA RTX A6000; funciona con cualquier GPU CUDA (o CPU,
mucho más lento).
