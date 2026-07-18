#!/usr/bin/env python3
"""CLI del sistema multi-agente proyecto-urbano-agentivo.

Comandos:
    python main.py manzanas                        # lista las claves cvegeo
    python main.py analizar --cvegeo <CVEGEO> [--imagenes DIR] [--satelite IMG]
    python main.py analizar --todas                # recalcula todas con datos
    python main.py riesgo                          # ranking de riesgo (BD)
    python main.py trafico [--loop]                # recolecta TomTom/histórico
    python main.py dashboard                       # lanza el Streamlit unificado
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


def imprimir_evento(evento: dict) -> None:
    etapa = evento.pop("etapa")
    evento.pop("ts", None)
    print(f"  ▸ [{etapa}] " + ", ".join(f"{k}={v}" for k, v in evento.items()))


def cmd_manzanas(_args) -> None:
    from src.agents import AgenteSIG
    for cv in AgenteSIG().listar_manzanas():
        print(cv)


def cmd_analizar(args) -> None:
    from src.agents import Orquestador
    orq = Orquestador()
    orq.suscribir(imprimir_evento)

    if args.todas:
        resultados = orq.analizar_todas_sync()
        print(f"\n✅ {len(resultados)} manzanas evaluadas.")
        return

    if not args.cvegeo:
        print("❌ Falta --cvegeo (o usa --todas).")
        sys.exit(1)

    imagenes = []
    if args.imagenes:
        carpeta = Path(args.imagenes)
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        imagenes = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in exts)
        print(f"📸 {len(imagenes)} imágenes de fachada encontradas en {carpeta}")

    resultado = orq.analizar_manzana_sync(
        args.cvegeo, imagenes_fachada=imagenes,
        imagen_satelital=args.satelite, conf=args.conf)
    print("\n══ Resultado ══")
    print(json.dumps(resultado["riesgo"], indent=2, ensure_ascii=False))


def cmd_riesgo(_args) -> None:
    from src.agents import AgenteRiesgo
    ranking = AgenteRiesgo().ranking()
    if not ranking:
        print("Sin manzanas evaluadas todavía — corre `python main.py analizar`.")
        return
    print(f"{'CVEGEO':<18} {'score':>6} {'congestión':>10} {'pisos':>6} "
          f"{'población':>10} {'fuente':>14}")
    for r in ranking:
        print(f"{r['cvegeo']:<18} {r['score_riesgo']:>6.3f} "
              f"{(r['congestion'] or 0):>9.0%} {(r['altura_promedio_pisos'] or 0):>6.1f} "
              f"{str(r['poblacion_estimada'] or '—'):>10} {str(r['fuente_congestion']):>14}")


def cmd_trafico(args) -> None:
    from src.agents import AgenteSIG
    from src.tools import trafico
    sig = AgenteSIG()
    if args.loop:
        trafico.recolectar_en_loop(intervalo_s=args.intervalo,
                                   buscar_manzana=sig.localizar)
    else:
        resultados = trafico.recolectar_trafico_sync(buscar_manzana=sig.localizar)
        for r in resultados:
            cong = f"{r['congestion']:.0%}" if r["congestion"] is not None else "s/d"
            print(f"  {r['vialidad']}: {cong} [{r['fuente']}] "
                  f"(manzana: {r['cvegeo'] or 'no identificada'})")
        print(f"✅ {len(resultados)} calles guardadas.")


def cmd_dashboard(_args) -> None:
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(RAIZ / "src" / "dashboard" / "app.py")])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sistema multi-agente de análisis urbano")
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("manzanas", help="Lista las claves CVEGEO disponibles")

    p_analizar = sub.add_parser("analizar", help="Pipeline completo SIG → Visión → Riesgo → BD")
    p_analizar.add_argument("--cvegeo", help="Clave de la manzana a analizar")
    p_analizar.add_argument("--todas", action="store_true",
                            help="Recalcula todas las manzanas con datos")
    p_analizar.add_argument("--imagenes", help="Carpeta con fotografías de fachada")
    p_analizar.add_argument("--satelite", help="Imagen satelital del área")
    p_analizar.add_argument("--conf", type=float, default=None,
                            help="Umbral de confianza (0.05–0.95)")

    sub.add_parser("riesgo", help="Ranking de riesgo por manzana (BD)")

    p_trafico = sub.add_parser("trafico", help="Recolecta congestión (TomTom → histórico)")
    p_trafico.add_argument("--loop", action="store_true")
    p_trafico.add_argument("--intervalo", type=int, default=1800)

    sub.add_parser("dashboard", help="Lanza el dashboard Streamlit unificado")

    args = parser.parse_args()
    {"manzanas": cmd_manzanas, "analizar": cmd_analizar, "riesgo": cmd_riesgo,
     "trafico": cmd_trafico, "dashboard": cmd_dashboard}[args.comando](args)


if __name__ == "__main__":
    main()
