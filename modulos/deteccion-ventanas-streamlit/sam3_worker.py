#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Worker de SAM3, pensado para correr con el Python de venv_sam3 (3.12 + torch 2.7),
invocado como subproceso desde app.py (que corre en el venv normal, con Detectron2/
YOLO en torch 2.1). Los dos entornos no pueden convivir en el mismo proceso.

Uso:
    python3 sam3_worker.py <prompt_texto> <umbral_conf> <ruta_img1> [<ruta_img2> ...]

Salida: un único JSON por stdout, con esta forma:
    {
      "<ruta_img1>": {"cajas": [{"box": [x1,y1,x2,y2], "score": 0.91}, ...]},
      "<ruta_img2>": {"error": "mensaje"},
      ...
    }
"""

import sys
import json


def main():
    prompt = sys.argv[1]
    conf = float(sys.argv[2])
    rutas = sys.argv[3:]

    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()
    if torch.cuda.is_available():
        model = model.to("cuda")
    processor = Sam3Processor(model)

    resultados = {}

    for ruta in rutas:
        try:
            imagen = Image.open(ruta).convert("RGB")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(imagen)
                salida = processor.set_text_prompt(state=state, prompt=prompt)

            boxes = salida["boxes"].float().cpu().numpy()
            scores = salida["scores"].float().cpu().numpy()

            cajas = [
                {"box": [float(v) for v in boxes[i]], "score": float(scores[i])}
                for i in range(len(boxes))
                if scores[i] >= conf
            ]
            resultados[ruta] = {"cajas": cajas}
        except Exception as e:
            resultados[ruta] = {"error": str(e)}

    print(json.dumps(resultados))


if __name__ == "__main__":
    main()
