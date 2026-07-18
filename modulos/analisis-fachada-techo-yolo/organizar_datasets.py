"""
Organiza los datasets nuevos ("datasets_fuente/Facade/" y "datasets_fuente/techos/")
en carpetas YOLO listas para entrenar, separadas por categoría para poder entrenar
"parte por parte", más una fusión "todo junto" para las categorías que sí son de fachada.

Categorías generadas:
  - datasets/dataset_techo_yolo/ Techos/edificios (clase única 'edificio')
  - datasets/dataset_ventanas/         Solo ventanas (clase única 'window')
  - datasets/dataset_fachada_general/  Elementos arquitectónicos de fachada (25 clases)
  - datasets/dataset_danos/            Grietas y daños estructurales (10 clases)
  - datasets/dataset_fachadas/         FUSIÓN de ventanas + fachada general + daños
                                  + puertas + árboles, con nombres de clase
                                  normalizados (17 clases) — es la que usa
                                  entrenar_fachada.py / el servidor.
  - datasets/dataset_senales/          Señalamiento vial (47 clases) — NO es fachada,
                                  se deja aparte como categoría independiente.
  - datasets/dataset_calles/           Escena de calle: autos, peatones, etc. (12 clases)
                                  — tampoco es fachada, categoría independiente.

Uso (siempre desde la raíz del proyecto):
    python3 organizar_datasets.py            # organiza de verdad
    python3 organizar_datasets.py --dry-run  # solo muestra qué haría
"""

import argparse
import hashlib
import json
import os
import re
import shutil

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
AREA_MIN_MASCARA = 30  # px^2 mínimos para que un componente conexo cuente como edificio


def nombre_seguro(tag, base):
    """Nombre corto y determinístico (tag + hash) para evitar el límite de 255
    caracteres del sistema de archivos con los nombres largos de Roboflow."""
    h = hashlib.md5(base.encode('utf-8')).hexdigest()[:16]
    return f"{tag}_{h}"

# ── Taxonomías objetivo ──
CLASES_TECHO = ['edificio']

CLASES_VENTANAS = ['window']

CLASES_FACHADA_GENERAL = [
    'Balcony', 'Blind', 'Building Outline', 'Corner Stone', 'Cornice', 'Deco',
    'Decorative Shell', 'Door', 'Door Hole', 'Facade', 'Floor', 'Floor Slab',
    'Floor Slabs', 'Glass Wall', 'Ground Floor', 'Molding', 'Pillar',
    'Raised Floor', 'Roof', 'Shop', 'Sill', 'Top Floor', 'Window',
    'Window Hole', 'Window Outline',
]

CLASES_DANOS = [
    'crack', 'ac_bracket_corrosion', 'concrete_spalling',
    'exposed_reinforcement', 'peeling_plaster', 'tile_detachment',
    'corrosion', 'delamination', 'dirty_mold', 'paint_defect',
]

# Fusión "todo junto" de fachada: taxonomía estructural + daño ya normalizada
# (nombres en minúsculas, coincide con lo que espera servidor_deteccion.py).
# 'door' y 'tree' se agregaron con los datasets nuevos — árboles detectados junto
# a la fachada no tienen su propia categoría en el menú, van como clase más aquí.
CLASES_FACHADA_MERGED = [
    'balcony', 'entrance', 'fence', 'floor', 'window', 'door', 'tree',
    'crack', 'ac_bracket_corrosion', 'concrete_spalling',
    'exposed_reinforcement', 'peeling_plaster', 'tile_detachment',
    'corrosion', 'delamination', 'dirty_mold', 'paint_defect',
]

CLASES_SENALES = [
    'ANIMALS', 'CONSTRUCTION', 'CYCLES CROSSING', 'DANGER', 'NO ENTRY',
    'PEDESTRIAN CROSSING', 'SCHOOL CROSSING', 'SNOW', 'STOP', 'bend',
    'bend left', 'bend right', 'building', 'give way', 'go left',
    'go left or straight', 'go right', 'go right or straight', 'go straight',
    'keep left', 'keep right', 'no overtaking', 'no overtaking -trucks-',
    'no traffic both ways', 'no trucks', 'priority at next intersection',
    'priority road', 'restriction ends', 'restriction ends -overtaking -trucks-',
    'restriction ends -overtaking-', 'restriction ends 80', 'road',
    'road narrows', 'roundabout', 'slippery road', 'speed limit 100',
    'speed limit 120', 'speed limit 20', 'speed limit 30', 'speed limit 50',
    'speed limit 60', 'speed limit 70', 'speed limit 80', 'traffic signal',
    'trafficlight', 'uneven road', 'window',
]

CLASES_CALLES = [
    'animal', 'auto', 'bike', 'bus', 'car', 'carrier_vehicle', 'driver',
    'num_plate', 'pedestrain', 'person', 'scooty', 'transport_stop',
]


def bbox_desde_valores(valores):
    """Acepta una línea de etiqueta YOLO ya sea en formato bbox (4 valores:
    xc,yc,w,h) o en formato de polígono YOLO-seg (N pares x,y intercalados,
    típico de varias exportaciones de Roboflow) y siempre devuelve un bbox
    normalizado (xc, yc, w, h). Con polígonos, se usa la caja envolvente
    (min/max de todos los puntos)."""
    if len(valores) == 4:
        return tuple(valores)
    if len(valores) >= 6 and len(valores) % 2 == 0:
        xs = valores[0::2]
        ys = valores[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return ((x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min)
    return None  # formato irreconocible, se descarta la línea


def remapear_yolo(carpeta_origen, splits_map, tag, mapa_clases, clases_objetivo,
                   dest_root, stats):
    """Copia+remapea un dataset YOLO (images/ + labels/ por split). Detecta y
    convierte automáticamente etiquetas en formato polígono (YOLO-seg) a bbox."""
    for split_origen, split_destino in splits_map.items():
        img_dir = os.path.join(RAIZ, carpeta_origen, split_origen, 'images')
        lbl_dir = os.path.join(RAIZ, carpeta_origen, split_origen, 'labels')
        if not os.path.isdir(img_dir):
            continue

        dst_img_dir = os.path.join(RAIZ, dest_root, split_destino, 'images')
        dst_lbl_dir = os.path.join(RAIZ, dest_root, split_destino, 'labels')
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        for fname in sorted(os.listdir(img_dir)):
            base, ext = os.path.splitext(fname)
            lbl_path = os.path.join(lbl_dir, base + '.txt')
            if not os.path.isfile(lbl_path):
                continue

            nuevo_nombre = nombre_seguro(tag, base)
            dst_lbl_path = os.path.join(dst_lbl_dir, nuevo_nombre + '.txt')
            if os.path.isfile(dst_lbl_path):
                stats['ya_procesadas'] += 1
                continue

            nuevas_lineas = []
            with open(lbl_path, 'r') as f:
                for linea in f:
                    partes = linea.split()
                    if not partes:
                        continue
                    cls_idx = int(partes[0])
                    nombre_clase = mapa_clases.get(cls_idx)
                    if nombre_clase is None:
                        continue
                    bbox = bbox_desde_valores([float(v) for v in partes[1:]])
                    if bbox is None:
                        continue
                    nuevo_idx = clases_objetivo.index(nombre_clase)
                    xc, yc, w, h = bbox
                    nuevas_lineas.append(f"{nuevo_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            if not nuevas_lineas:
                continue

            stats['imagenes'] += 1
            stats['cajas'] += len(nuevas_lineas)
            if stats['dry_run']:
                continue

            shutil.copy2(os.path.join(img_dir, fname), os.path.join(dst_img_dir, nuevo_nombre + ext))
            with open(dst_lbl_path, 'w') as f:
                f.write('\n'.join(nuevas_lineas) + '\n')


def coco_a_yolo_techo(images_dir, coco_json_path, dest_root, split_destino, tag, stats):
    """Convierte anotaciones COCO (categoría única 'building') a YOLO clase 'edificio'."""
    if not os.path.isfile(coco_json_path):
        return
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    id_a_nombre = {img['id']: img['file_name'] for img in coco['images']}
    id_a_size = {img['id']: (img['width'], img['height']) for img in coco['images']}
    anotaciones_por_imagen = {}
    for ann in coco['annotations']:
        anotaciones_por_imagen.setdefault(ann['image_id'], []).append(ann)

    dst_img_dir = os.path.join(RAIZ, dest_root, split_destino, 'images')
    dst_lbl_dir = os.path.join(RAIZ, dest_root, split_destino, 'labels')
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    for image_id, anns in anotaciones_por_imagen.items():
        filename = id_a_nombre[image_id]
        img_w, img_h = id_a_size[image_id]
        base, ext = os.path.splitext(filename)
        nuevo_nombre = nombre_seguro(tag, base)

        dst_lbl_path = os.path.join(dst_lbl_dir, nuevo_nombre + '.txt')
        if os.path.isfile(dst_lbl_path):
            stats['ya_procesadas'] += 1
            continue

        src_img_path = os.path.join(RAIZ, images_dir, filename)
        if not os.path.isfile(src_img_path):
            continue

        lineas = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            xc = (x + w / 2) / img_w
            yc = (y + h / 2) / img_h
            lineas.append(f"0 {xc:.6f} {yc:.6f} {w/img_w:.6f} {h/img_h:.6f}")

        if not lineas:
            continue

        stats['imagenes'] += 1
        stats['cajas'] += len(lineas)
        if stats['dry_run']:
            continue

        shutil.copy2(src_img_path, os.path.join(dst_img_dir, nuevo_nombre + ext))
        with open(dst_lbl_path, 'w') as f:
            f.write('\n'.join(lineas) + '\n')


def mascara_a_yolo_techo(images_dir, mascaras_dir, dest_root, split_destino, tag, stats):
    """Convierte máscaras binarias (edificio=255) a cajas YOLO vía componentes conexos."""
    img_dir = os.path.join(RAIZ, images_dir)
    msk_dir = os.path.join(RAIZ, mascaras_dir)
    if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
        return

    dst_img_dir = os.path.join(RAIZ, dest_root, split_destino, 'images')
    dst_lbl_dir = os.path.join(RAIZ, dest_root, split_destino, 'labels')
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    for fname in sorted(os.listdir(img_dir)):
        mask_path = os.path.join(msk_dir, fname)
        if not os.path.isfile(mask_path):
            continue

        base, ext = os.path.splitext(fname)
        nuevo_nombre = nombre_seguro(tag, base)
        dst_lbl_path = os.path.join(dst_lbl_dir, nuevo_nombre + '.txt')
        if os.path.isfile(dst_lbl_path):
            stats['ya_procesadas'] += 1
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape
        _, binaria = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        n, _, stats_cc, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)

        lineas = []
        for i in range(1, n):
            x, y, bw, bh, area = stats_cc[i]
            if area < AREA_MIN_MASCARA:
                continue
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            lineas.append(f"0 {xc:.6f} {yc:.6f} {bw/w:.6f} {bh/h:.6f}")

        if not lineas:
            continue

        stats['imagenes'] += 1
        stats['cajas'] += len(lineas)
        if stats['dry_run']:
            continue

        shutil.copy2(os.path.join(img_dir, fname), os.path.join(dst_img_dir, nuevo_nombre + ext))
        with open(dst_lbl_path, 'w') as f:
            f.write('\n'.join(lineas) + '\n')


def cmp_xml_a_yolo(carpeta_base, mapa_clases, clases_objetivo, dest_root, tag, stats):
    """Convierte el formato de CMP_facade_DB_base: un .xml por imagen (mismo nombre
    base que el .jpg), con cajas YA normalizadas 0-1 como
    <object><points><x>..</x><x>..</x><y>..</y><y>..</y></points><labelname>..</labelname></object>
    — no hay que leer tamaño de imagen ni convertir escala, solo tomar min/max de
    cada par x/y. No trae splits propios (es una sola carpeta 'base'), así que se
    reparten aquí mismo ~85/10/5 train/valid/test, determinístico por orden alfabético
    de archivo para que sea reproducible entre corridas."""
    base_dir = os.path.join(RAIZ, carpeta_base)
    if not os.path.isdir(base_dir):
        return

    for i, xml_name in enumerate(sorted(f for f in os.listdir(base_dir) if f.endswith('.xml'))):
        base = xml_name[:-4]
        img_path = os.path.join(base_dir, base + '.jpg')
        if not os.path.isfile(img_path):
            continue

        if i % 20 == 0:
            split = 'test'
        elif i % 10 == 0:
            split = 'valid'
        else:
            split = 'train'

        nuevo_nombre = nombre_seguro(tag, base)
        dst_img_dir = os.path.join(RAIZ, dest_root, split, 'images')
        dst_lbl_dir = os.path.join(RAIZ, dest_root, split, 'labels')
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        dst_lbl_path = os.path.join(dst_lbl_dir, nuevo_nombre + '.txt')
        if os.path.isfile(dst_lbl_path):
            stats['ya_procesadas'] += 1
            continue

        with open(os.path.join(base_dir, xml_name), encoding='utf-8', errors='ignore') as f:
            contenido = f.read()

        lineas = []
        for obj in re.finditer(r'<object>(.*?)</object>', contenido, re.S):
            bloque = obj.group(1)
            xs = re.findall(r'<x>\s*([-\d.]+)\s*</x>', bloque)
            ys = re.findall(r'<y>\s*([-\d.]+)\s*</y>', bloque)
            nombre_m = re.search(r'<labelname>\s*([^<\s][^<]*?)\s*</labelname>', bloque)
            if len(xs) != 2 or len(ys) != 2 or not nombre_m:
                continue  # forma no rectangular o mal formada, se descarta
            nombre_clase = mapa_clases.get(nombre_m.group(1))
            if nombre_clase is None:
                continue
            x1, x2 = sorted(float(v) for v in xs)
            y1, y2 = sorted(float(v) for v in ys)
            xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            nuevo_idx = clases_objetivo.index(nombre_clase)
            lineas.append(f"{nuevo_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not lineas:
            continue

        stats['imagenes'] += 1
        stats['cajas'] += len(lineas)
        if stats['dry_run']:
            continue

        shutil.copy2(img_path, os.path.join(dst_img_dir, nuevo_nombre + '.jpg'))
        with open(dst_lbl_path, 'w') as f:
            f.write('\n'.join(lineas) + '\n')


def nuevas_stats(dry_run):
    return {'imagenes': 0, 'cajas': 0, 'ya_procesadas': 0, 'dry_run': dry_run}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    dry = args.dry_run

    reporte = {}

    # ═══════ TECHO ═══════
    print("🔄 Organizando TECHO...")
    reporte['techo'] = []

    s = nuevas_stats(dry)
    for split_o, split_d, jsonf in [
        ('2.1 train/train', 'train', '2.4 annotation/annotation/train.json'),
        ('2.2 test/test', 'test', '2.4 annotation/annotation/test.json'),
        ('2.3 valid/validation', 'val', '2.4 annotation/annotation/validation.json'),
    ]:
        coco_a_yolo_techo(
            f'datasets_fuente/techos/Aerial imagery dataset/{split_o}',
            os.path.join(RAIZ, f'datasets_fuente/techos/Aerial imagery dataset/{jsonf}'),
            'datasets/dataset_techo_yolo', split_d, 'aerial', s,
        )
    reporte['techo'].append({'fuente': 'Aerial imagery dataset (COCO)', **s, 'dry_run': None})
    del reporte['techo'][-1]['dry_run']

    s = nuevas_stats(dry)
    mascara_a_yolo_techo(
        'datasets_fuente/techos/Satellite dataset I (global cities)/Satellite dataset Ⅰ (global cities)/Satellite dataset ó± (global cities)/image',
        'datasets_fuente/techos/Satellite dataset I (global cities)/Satellite dataset Ⅰ (global cities)/Satellite dataset ó± (global cities)/label',
        'datasets/dataset_techo_yolo', 'train', 'satI', s,
    )
    reporte['techo'].append({'fuente': 'Satellite dataset I (global cities) [máscara]', **s, 'dry_run': None})
    del reporte['techo'][-1]['dry_run']

    s = nuevas_stats(dry)
    mascara_a_yolo_techo(
        'datasets_fuente/techos/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/test/image',
        'datasets_fuente/techos/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/test/label',
        'datasets/dataset_techo_yolo', 'train', 'satII', s,
    )
    reporte['techo'].append({'fuente': 'Satellite dataset II (East Asia) [máscara]', **s, 'dry_run': None})
    del reporte['techo'][-1]['dry_run']

    # 'train' de Satellite II no existía en la descarga anterior (solo 'test'); se
    # completó con la carpeta 'nuevos/' — mucho más grande (3135 imágenes vs 903).
    s = nuevas_stats(dry)
    mascara_a_yolo_techo(
        'datasets_fuente/techos/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/image',
        'datasets_fuente/techos/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/label',
        'datasets/dataset_techo_yolo', 'train', 'satIItrain', s,
    )
    reporte['techo'].append({'fuente': 'Satellite dataset II (East Asia) [máscara, train]', **s, 'dry_run': None})
    del reporte['techo'][-1]['dry_run']

    fuentes_techo_yolo = [
        ('datasets_fuente/techos/ROOF.v5i.yolov11', 'roofv5', {0: 'edificio', 1: 'edificio'}),
        ('datasets_fuente/techos/roof.v1i.yolov11', 'roofv1', {0: 'edificio', 1: 'edificio', 2: 'edificio', 3: 'edificio'}),
        ('datasets_fuente/techos/Cool Roof Detection.v1i.yolov11', 'coolroof', {0: 'edificio', 1: 'edificio'}),  # 2='object' se descarta
        ('datasets_fuente/techos/Satellite.v4i.yolov11', 'satv4', {5: 'edificio', 6: 'edificio', 7: 'edificio', 8: 'edificio', 9: 'edificio', 10: 'edificio', 11: 'edificio', 12: 'edificio'}),
        # ── Datasets nuevos (satelital, todas confirmadas visualmente como aéreas) ──
        ('datasets_fuente/techos/buildings-detection-5-classes.v21i.yolov11', 'build5c',
         {0: 'edificio', 1: 'edificio', 2: 'edificio', 3: 'edificio', 4: 'edificio'}),  # Business/Immeuble/Industrielle/Maison/Villa: todos son tipos de edificio
        ('datasets_fuente/techos/Buildings.v1i.yolov11', 'buildx640', {0: 'edificio'}),  # 1='object' (10 cajas, ruido) se descarta
        ('datasets_fuente/techos/Buildings.v1i.yolov11 (1)', 'builde7kcx', {0: 'edificio'}),
        ('datasets_fuente/techos/buildings detection 500m.v1-buildings_detection_10k.yolov11', 'build500m',
         {0: 'edificio', 1: 'edificio', 2: 'edificio', 3: 'edificio', 4: 'edificio'}),  # immeuble/maison/other/social/villa
    ]
    for carpeta, tag, mapa in fuentes_techo_yolo:
        s = nuevas_stats(dry)
        remapear_yolo(carpeta, {'train': 'train', 'valid': 'val', 'test': 'test'}, tag, mapa, CLASES_TECHO, 'datasets/dataset_techo_yolo', s)
        reporte['techo'].append({'fuente': carpeta, **s, 'dry_run': None})
        del reporte['techo'][-1]['dry_run']

    # ═══════ VENTANAS (solo window) ═══════
    print("🔄 Organizando VENTANAS...")
    reporte['ventanas'] = []
    fuentes_ventanas = [
        ('datasets_fuente/Facade/windows/facade.v4i.yolov11', 'facadev4', {4: 'window'}),
        ('datasets_fuente/Facade/windows/peepshield-facade.v1i.yolov11', 'peepshield', {1: 'window'}),
        ('datasets_fuente/Facade/windows/Facade.v1i.yolov11', 'facadev1win', {0: 'window'}),
    ]
    for carpeta, tag, mapa in fuentes_ventanas:
        s = nuevas_stats(dry)
        remapear_yolo(carpeta, {'train': 'train', 'valid': 'valid', 'test': 'test'}, tag, mapa, CLASES_VENTANAS, 'datasets/dataset_ventanas', s)
        reporte['ventanas'].append({'fuente': carpeta, **s, 'dry_run': None})
        del reporte['ventanas'][-1]['dry_run']

    # ═══════ FACHADA GENERAL (elementos arquitectónicos) ═══════
    print("🔄 Organizando FACHADA GENERAL...")
    reporte['fachada_general'] = []
    mapa_general = {i: nombre for i, nombre in enumerate([
        'Balcony', 'Blind', 'Building Outline', 'Corner Stone', 'Cornice', 'Deco',
        'Decorative Shell', None, 'Door', 'Door Hole', 'Facade', 'Floor', 'Floor Slab',
        'Floor Slabs', 'Glass Wall', 'Ground Floor', 'Molding', 'Pillar',
        'Raised Floor', 'Roof', 'Shop', 'Sill', 'Top Floor', 'Window',
        'Window Hole', 'Window Outline', None, None,
    ])}
    s = nuevas_stats(dry)
    remapear_yolo('datasets_fuente/Facade/facade.v1i.yolov11', {'train': 'train', 'valid': 'valid', 'test': 'test'},
                  'facadegen', mapa_general, CLASES_FACHADA_GENERAL, 'datasets/dataset_fachada_general', s)
    reporte['fachada_general'].append({'fuente': 'datasets_fuente/Facade/facade.v1i.yolov11', **s, 'dry_run': None})
    del reporte['fachada_general'][-1]['dry_run']

    # CMP_facade_DB_base: 378 fachadas con cajas XML ya normalizadas (formato propio,
    # no YOLO — ver cmp_xml_a_yolo). Sus 11 clases mapean 1:1 con nombres ya existentes
    # en CLASES_FACHADA_GENERAL, así que no hace falta agregar clases nuevas.
    # (El dataset "pix2pix-facades-dataset" de Kaggle es la MISMA base CMP —confirmado:
    # 606 imágenes = 378 "base" + 228 "extended", mismo origen— pero repackeada como
    # pares foto+máscara de color para pix2pix, sin cajas. Extraer cajas de esa máscara
    # implicaría segmentar colores JPEG-comprimidos y ruidosos por conexión de componentes,
    # bastante menos preciso que las cajas XML que ya vienen limpias aquí, así que no se usa.)
    mapa_cmp_general = {
        'facade': 'Facade', 'window': 'Window', 'door': 'Door', 'cornice': 'Cornice',
        'sill': 'Sill', 'balcony': 'Balcony', 'blind': 'Blind', 'deco': 'Deco',
        'molding': 'Molding', 'pillar': 'Pillar', 'shop': 'Shop',
    }
    s = nuevas_stats(dry)
    cmp_xml_a_yolo('CMP_facade_DB_base/base', mapa_cmp_general, CLASES_FACHADA_GENERAL,
                    'datasets/dataset_fachada_general', 'cmpbase', s)
    reporte['fachada_general'].append({'fuente': 'CMP_facade_DB_base (XML bbox)', **s, 'dry_run': None})
    del reporte['fachada_general'][-1]['dry_run']

    # ═══════ DAÑOS ═══════
    print("🔄 Organizando DAÑOS...")
    reporte['danos'] = []
    fuentes_danos = [
        ('datasets_fuente/Facade/Daños-Facade/Facade Cracks.v3-v03.yolov11', 'cracks', {0: 'crack'}),
        ('datasets_fuente/Facade/Daños-Facade/external facade.v2i.yolov11', 'external',
         {0: 'ac_bracket_corrosion', 1: 'concrete_spalling', 2: 'exposed_reinforcement',
          3: 'peeling_plaster', 4: 'tile_detachment', 5: 'crack'}),
        # nc=6: corrosion, crack, delamination, dirty- mold, paint defect, rust
        # 'corrosion' y 'rust' se fusionan en la misma clase genérica (óxido/corrosión de metal)
        ('datasets_fuente/Facade/Daños-Facade/Defects in Facade Building.v1i.yolov11', 'defectsfacade',
         {0: 'corrosion', 1: 'crack', 2: 'delamination', 3: 'dirty_mold', 4: 'paint_defect', 5: 'corrosion'}),
    ]
    for carpeta, tag, mapa in fuentes_danos:
        s = nuevas_stats(dry)
        remapear_yolo(carpeta, {'train': 'train', 'valid': 'valid', 'test': 'test'}, tag, mapa, CLASES_DANOS, 'datasets/dataset_danos', s)
        reporte['danos'].append({'fuente': carpeta, **s, 'dry_run': None})
        del reporte['danos'][-1]['dry_run']

    # ═══════ FACHADA FUSIONADA ("todo junto") ═══════
    # entrenar_fachada.py ya terminó su entrenamiento, así que 'dataset_fachadas'
    # (destino) se movió a datasets/ junto con las demás categorías procesadas.
    print("🔄 Organizando FACHADA FUSIONADA (ventanas + general + daños)...")
    reporte['fachada_fusionada'] = []
    fuentes_fusion = [
        ('datasets_fuente/Facade/windows/facade.v4i.yolov11', 'facadev4',
         {0: 'balcony', 1: 'entrance', 2: 'fence', 3: 'floor', 4: 'window'}),
        ('datasets_fuente/Facade/windows/peepshield-facade.v1i.yolov11', 'peepshield',
         {0: 'entrance', 1: 'window'}),
        ('datasets_fuente/Facade/windows/Facade.v1i.yolov11', 'facadev1win', {0: 'window'}),
        ('datasets_fuente/Facade/facade.v1i.yolov11', 'facadegen',
         {0: 'balcony', 8: 'entrance', 9: 'entrance', 11: 'floor', 12: 'floor',
          13: 'floor', 15: 'floor', 18: 'floor', 22: 'floor', 23: 'window',
          24: 'window', 25: 'window'}),
        ('datasets_fuente/Facade/Daños-Facade/Facade Cracks.v3-v03.yolov11', 'cracks', {0: 'crack'}),
        ('datasets_fuente/Facade/Daños-Facade/external facade.v2i.yolov11', 'external',
         {0: 'ac_bracket_corrosion', 1: 'concrete_spalling', 2: 'exposed_reinforcement',
          3: 'peeling_plaster', 4: 'tile_detachment', 5: 'crack'}),
        ('datasets_fuente/Facade/Daños-Facade/Defects in Facade Building.v1i.yolov11', 'defectsfacade',
         {0: 'corrosion', 1: 'crack', 2: 'delamination', 3: 'dirty_mold', 4: 'paint_defect', 5: 'corrosion'}),
        # nc=4: Building Facade, Door, Entrance, Window — 'Building Facade' (10 cajas,
        # bbox genérico de toda la fachada) se descarta por ser ruido, no un elemento puntual
        ('datasets_fuente/Facade/Facade Factors Finder.v2i.yolov11', 'facadefactors',
         {1: 'door', 2: 'entrance', 3: 'window'}),
        # Árboles: sin categoría propia en el menú, se detectan como clase más del
        # modelo de fachada. 'giesta'/'Trees' (clase 0 en ambas fuentes) se descarta —
        # es una especie de arbusto distinta, no un árbol genérico.
        ('datasets_fuente/Facade/trees/trees.v5-simpletreegiestasqual.yolov11', 'treesv5', {1: 'tree'}),
        ('datasets_fuente/Facade/trees/Trees.v1i.yolov11', 'treesv1', {1: 'tree'}),
    ]
    for carpeta, tag, mapa in fuentes_fusion:
        s = nuevas_stats(dry)
        remapear_yolo(carpeta, {'train': 'train', 'valid': 'valid', 'test': 'test'}, tag, mapa, CLASES_FACHADA_MERGED, 'datasets/dataset_fachadas', s)
        reporte['fachada_fusionada'].append({'fuente': carpeta, **s, 'dry_run': None})
        del reporte['fachada_fusionada'][-1]['dry_run']

    # CMP_facade_DB_base también aporta a la fusión, pero solo las 3 clases que
    # existen en la taxonomía fusionada (balcony/door/window) — cornice/sill/blind/
    # deco/molding/pillar/shop/facade no tienen equivalente ahí, se quedan solo en
    # dataset_fachada_general (arriba).
    mapa_cmp_merged = {'window': 'window', 'door': 'door', 'balcony': 'balcony'}
    s = nuevas_stats(dry)
    cmp_xml_a_yolo('CMP_facade_DB_base/base', mapa_cmp_merged, CLASES_FACHADA_MERGED,
                    'datasets/dataset_fachadas', 'cmpbasemerged', s)
    reporte['fachada_fusionada'].append({'fuente': 'CMP_facade_DB_base (XML bbox) [balcony/door/window]', **s, 'dry_run': None})
    del reporte['fachada_fusionada'][-1]['dry_run']

    # ═══════ SEÑALES (independiente, no es fachada) ═══════
    print("🔄 Organizando SEÑALES...")
    reporte['senales'] = []
    mapa_senales = {i: nombre for i, nombre in enumerate(CLASES_SENALES)}
    s = nuevas_stats(dry)
    remapear_yolo('datasets_fuente/Facade/Señales/Facade.v2i.yolov11', {'train': 'train', 'valid': 'valid', 'test': 'test'},
                  'senales', mapa_senales, CLASES_SENALES, 'datasets/dataset_senales', s)
    reporte['senales'].append({'fuente': 'datasets_fuente/Facade/Señales/Facade.v2i.yolov11', **s, 'dry_run': None})
    del reporte['senales'][-1]['dry_run']

    # 'Road signs.v1i' (holandés: Stop-bord/voorrangweg/zebrapad) y 'road signs.v1-release-640'
    # (inglés, 21 clases de señalamiento vial + semáforo) — de esta última solo se mapean las
    # clases con equivalente claro en CLASES_SENALES; el resto (do_not_turn_l, parking,
    # railway_crossing, colores de semáforo por separado, etc.) se descarta por no tener
    # una clase análoga ya existente, en vez de forzar un mapeo aproximado.
    fuentes_senales_nuevas = [
        ('datasets_fuente/Facade/Señales/Road signs.v1i.yolov11', 'roadsignsnl',
         {0: 'STOP', 1: 'priority road', 2: 'PEDESTRIAN CROSSING'}),
        ('datasets_fuente/Facade/Señales/road signs.v1-release-640.yolov11', 'roadsigns640', {
            1: 'NO ENTRY', 11: 'PEDESTRIAN CROSSING', 12: 'PEDESTRIAN CROSSING',
            15: 'STOP', 17: 'trafficlight', 19: 'DANGER',
        }),
    ]
    for carpeta, tag, mapa in fuentes_senales_nuevas:
        s = nuevas_stats(dry)
        remapear_yolo(carpeta, {'train': 'train', 'valid': 'valid', 'test': 'test'}, tag, mapa, CLASES_SENALES, 'datasets/dataset_senales', s)
        reporte['senales'].append({'fuente': carpeta, **s, 'dry_run': None})
        del reporte['senales'][-1]['dry_run']

    # ═══════ CALLES (independiente, no es fachada) ═══════
    print("🔄 Organizando CALLES...")
    reporte['calles'] = []
    fuentes_calles = [
        ('datasets_fuente/Facade/streets/Streets.v3i.yolov11', 'streetsv3', {i: nombre for i, nombre in enumerate([
            'animal', 'auto', 'bike', 'bus', 'car', 'carrier_vehicle', 'driver',
            'num_plate', 'pedestrain', 'person', 'scooty'])}),
        ('datasets_fuente/Facade/streets/streets.v1i.yolov11', 'streetsv1', {0: 'bus', 1: 'car', 2: 'transport_stop'}),
        # 1 sola clase 'people' (378 img de "base" no, aquí son 30 img — dataset chico pero limpio)
        ('datasets_fuente/Facade/streets/people.v1-roboflow-instant-1--eval-.yolov11', 'peoplev1', {0: 'person'}),
        # nc=5: '0' (placeholder sin cajas reales, se descarta), Bus, Motorcycle, car, truck.
        # 'Motorcycle' se mapea a 'bike' (no hay clase específica de motocicleta en la taxonomía)
        # y 'truck' a 'carrier_vehicle' (vehículo de carga, el equivalente más cercano ya existente).
        ('datasets_fuente/Facade/streets/Vehicles.v2i.yolov11', 'vehiclesv2',
         {1: 'bus', 2: 'bike', 3: 'car', 4: 'carrier_vehicle'}),
    ]
    for carpeta, tag, mapa in fuentes_calles:
        s = nuevas_stats(dry)
        remapear_yolo(carpeta, {'train': 'train', 'valid': 'valid', 'test': 'test'}, tag, mapa, CLASES_CALLES, 'datasets/dataset_calles', s)
        reporte['calles'].append({'fuente': carpeta, **s, 'dry_run': None})
        del reporte['calles'][-1]['dry_run']

    # ═══════ Resumen ═══════
    print("\n=== Resumen ===")
    for categoria, resumen in reporte.items():
        print(f"\n{categoria.upper()}:")
        for r in resumen:
            extra = f" ({r['ya_procesadas']} ya procesadas)" if r['ya_procesadas'] else ""
            print(f"  - {r['fuente']}: {r['imagenes']} imágenes, {r['cajas']} cajas{extra}")

    reporte_path = os.path.join(RAIZ, 'reporte_organizacion_datasets.json')
    with open(reporte_path, 'w') as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Reporte guardado en {reporte_path}")

    if dry:
        print("\n(Modo --dry-run: no se copió ningún archivo)")
    else:
        print("\n✅ Organización completada.")


if __name__ == '__main__':
    main()
