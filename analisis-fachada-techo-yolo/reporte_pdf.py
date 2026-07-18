"""
Generación del reporte PDF de resultados de detección (techos + fachadas).
Sustituye a la antigua exportación en GeoJSON: el usuario ahora descarga un
PDF con métricas, gráficas y la tabla de datos por manzana.
"""

import io
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # sin display, solo renderizado a imagen
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

ACCENT = colors.HexColor('#c65a2e')
INK = colors.HexColor('#1a1a1a')
INK_SOFT = colors.HexColor('#6b6b6b')
LINE = colors.HexColor('#d9d9d6')


def _grafica_pisos_png(floor_distribution):
    """Renderiza la distribución de edificios por número de pisos como PNG en memoria."""
    if not floor_distribution:
        return None

    keys = sorted(floor_distribution.keys(), key=lambda k: int(k))
    values = [floor_distribution[k] for k in keys]
    labels = [f"{k} piso{'s' if k != '1' else ''}" for k in keys]

    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    ax.bar(labels, values, color='#c65a2e')
    ax.set_title('Distribución de edificios por número de pisos', fontsize=10)
    ax.tick_params(axis='x', labelsize=8, rotation=30)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf


def generar_pdf_reporte(total_buildings_analyzed, total_windows_detected,
                         total_registros, total_ventanas,
                         floor_distribution, manzanas_rows,
                         total_danos=0, damage_breakdown=None, trafico_rows=None):
    """
    Construye el PDF y lo devuelve como BytesIO listo para send_file().

    manzanas_rows: lista de tuplas (cvegeo, avg_pisos, max_pisos, total_ventanas, num_fotos, avg_altura)
    damage_breakdown: dict {nombre_clase_dano: cantidad_total}, ej. {'crack': 12, 'rust': 4}
    trafico_rows: lista de tuplas (cvegeo, congestion_promedio, num_lecturas) — de trafico_tomtom.py
    """
    damage_breakdown = damage_breakdown or {}
    trafico_rows = trafico_rows or []
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    titulo_style = ParagraphStyle('Titulo', parent=styles['Title'], textColor=INK, fontSize=18)
    subtitulo_style = ParagraphStyle('Subtitulo', parent=styles['Normal'], textColor=INK_SOFT, fontSize=9)
    seccion_style = ParagraphStyle('Seccion', parent=styles['Heading2'], textColor=ACCENT, fontSize=12, spaceBefore=16)

    elementos = []

    elementos.append(Paragraph('Reporte de Detección de Techos y Fachadas', titulo_style))
    elementos.append(Paragraph(
        f"Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}", subtitulo_style
    ))
    elementos.append(Spacer(1, 16))

    # ── Métricas globales ──
    elementos.append(Paragraph('Métricas globales', seccion_style))
    metricas_data = [
        ['Edificios analizados', str(total_buildings_analyzed)],
        ['Ventanas detectadas', str(total_windows_detected)],
        ['Daños detectados (grietas/defectos)', str(total_danos)],
        ['Registros en base de datos', str(total_registros)],
        ['Total de ventanas (BD)', str(total_ventanas)],
        ['Manzanas con datos', str(len(manzanas_rows))],
    ]
    tabla_metricas = Table(metricas_data, colWidths=[9*cm, 6*cm])
    tabla_metricas.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), INK_SOFT),
        ('TEXTCOLOR', (1, 0), (1, -1), INK),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, LINE),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elementos.append(tabla_metricas)

    # ── Gráfica de distribución de pisos ──
    grafica_buf = _grafica_pisos_png(floor_distribution)
    if grafica_buf:
        elementos.append(Paragraph('Distribución por número de pisos', seccion_style))
        elementos.append(RLImage(grafica_buf, width=16*cm, height=8*cm))
    else:
        elementos.append(Paragraph('Distribución por número de pisos', seccion_style))
        elementos.append(Paragraph('Aún no hay suficientes datos de fachadas para graficar.', styles['Normal']))

    # ── Desglose de daños detectados ──
    elementos.append(Paragraph('Daños y deterioro detectados', seccion_style))
    if damage_breakdown:
        NOMBRES_LEGIBLES = {
            'crack': 'Grietas', 'ac_bracket_corrosion': 'Corrosión en soporte de A/C',
            'concrete_spalling': 'Desprendimiento de concreto',
            'exposed_reinforcement': 'Varilla expuesta',
            'peeling_plaster': 'Aplanado descascarado', 'tile_detachment': 'Azulejo desprendido',
            'corrosion': 'Corrosión/óxido', 'delamination': 'Delaminación de recubrimiento',
            'dirty_mold': 'Suciedad/moho', 'paint_defect': 'Defecto de pintura',
        }
        filas_danos = [['Tipo de daño', 'Cantidad detectada']]
        for clase, cantidad in sorted(damage_breakdown.items(), key=lambda x: -x[1]):
            filas_danos.append([NOMBRES_LEGIBLES.get(clase, clase), str(cantidad)])
        tabla_danos = Table(filas_danos, colWidths=[9*cm, 6*cm])
        tabla_danos.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#b3261e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.4, LINE),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elementos.append(tabla_danos)
    else:
        elementos.append(Paragraph('No se han detectado daños en las fachadas analizadas.', styles['Normal']))

    # ── Congestión vial por manzana (trafico_tomtom.py) ──
    elementos.append(Paragraph('Congestión vial por manzana', seccion_style))
    if trafico_rows:
        filas_trafico = [['CVEGEO', 'Congestión promedio', 'Lecturas']]
        for cvegeo, congestion, num_lecturas in trafico_rows[:40]:
            filas_trafico.append([cvegeo, f"{congestion*100:.0f}%", str(num_lecturas)])
        tabla_trafico = Table(filas_trafico, repeatRows=1, colWidths=[6*cm, 5*cm, 4*cm])
        tabla_trafico.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), INK),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.4, LINE),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fafaf9')]),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elementos.append(tabla_trafico)
    else:
        elementos.append(Paragraph(
            'Sin datos de tráfico todavía. Corre trafico_tomtom.py (requiere TOMTOM_API_KEY) para recolectarlos.',
            styles['Normal']
        ))

    # ── Tabla de datos por manzana ──
    elementos.append(Paragraph('Datos por manzana (CVEGEO)', seccion_style))
    if manzanas_rows:
        encabezado = ['CVEGEO', 'Prom. pisos', 'Máx. pisos', 'Total ventanas', 'Fotos', 'Altura prom. (m)']
        filas = [encabezado] + [
            [row[0], row[1], row[2], row[3], row[4], row[5]]
            for row in manzanas_rows[:40]  # limitar a 40 filas para no desbordar el documento
        ]
        tabla_manzanas = Table(filas, repeatRows=1, colWidths=[3.2*cm, 2.6*cm, 2.6*cm, 3*cm, 2*cm, 3.2*cm])
        tabla_manzanas.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, 0), INK),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.4, LINE),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fafaf9')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elementos.append(tabla_manzanas)
        if len(manzanas_rows) > 40:
            elementos.append(Spacer(1, 6))
            elementos.append(Paragraph(
                f"Mostrando 40 de {len(manzanas_rows)} manzanas con datos (ordenadas por número de fotos analizadas).",
                subtitulo_style
            ))
    else:
        elementos.append(Paragraph('Aún no hay detecciones geolocalizadas guardadas.', styles['Normal']))

    doc.build(elementos)
    buffer.seek(0)
    return buffer
