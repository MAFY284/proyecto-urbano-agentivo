#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# Instalador automatizado de la Plataforma de Análisis Urbano
#
# 1. Crea el venv principal (./venv) e instala requirements.txt
# 2. Opcionalmente compila Detectron2 desde el repo oficial
# 3. Crea el SEGUNDO entorno aislado src/tools/env_sam3 (Python 3.12 +
#    PyTorch 2.7 + sam3) exclusivamente para la segmentación zero-shot
#    por texto — sus versiones de PyTorch son incompatibles con el core
#    YOLO11/Detectron2 del sistema principal, por eso NUNCA se mezclan.
#
# Uso:
#   bash install.sh              # venv principal solamente
#   bash install.sh --detectron2 # + compila Detectron2
#   bash install.sh --sam3       # + crea el venv aislado de SAM3
#   bash install.sh --todo       # todo lo anterior
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SAM3="$RAIZ/src/tools/env_sam3"

CON_DETECTRON2=false
CON_SAM3=false
for arg in "$@"; do
    case "$arg" in
        --detectron2) CON_DETECTRON2=true ;;
        --sam3)       CON_SAM3=true ;;
        --todo)       CON_DETECTRON2=true; CON_SAM3=true ;;
    esac
done

echo "══ [1/3] Entorno principal ══"
if [ ! -d "$RAIZ/venv" ]; then
    python3 -m venv "$RAIZ/venv"
fi
"$RAIZ/venv/bin/pip" install --upgrade pip
"$RAIZ/venv/bin/pip" install -r "$RAIZ/requirements.txt"

if $CON_DETECTRON2; then
    echo "══ [2/3] Detectron2 (compilación desde el repo oficial) ══"
    "$RAIZ/venv/bin/pip" install 'git+https://github.com/facebookresearch/detectron2.git' \
        || echo "⚠️  Detectron2 no se pudo compilar — el sistema funciona sin él (el motor queda oculto)."
else
    echo "══ [2/3] Detectron2 omitido (usa --detectron2 para instalarlo) ══"
fi

if $CON_SAM3; then
    echo "══ [3/3] Entorno aislado de SAM3 en $ENV_SAM3 ══"
    # REGLA CRÍTICA: SAM3 exige Python 3.12 + torch 2.7, incompatibles con el
    # entorno principal. Vive en su propio venv y solo se invoca por subproceso
    # (ver src/tools/comparador.py → sam3_worker.py).
    # Si no hay python3.12 del sistema, se usa uv (descarga un CPython gestionado).
    if [ ! -d "$ENV_SAM3" ]; then
        if command -v python3.12 >/dev/null 2>&1; then
            python3.12 -m venv "$ENV_SAM3"
        else
            command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
            uv venv --python 3.12 --seed "$ENV_SAM3"
        fi
    fi
    "$ENV_SAM3/bin/pip" install --upgrade pip
    "$ENV_SAM3/bin/pip" install 'torch==2.7.*' torchvision
    # sam3 (repo de Meta) + dependencias que su setup no declara completas.
    # setuptools<81: sam3 usa pkg_resources, eliminado en setuptools>=83.
    "$ENV_SAM3/bin/pip" install 'git+https://github.com/facebookresearch/sam3.git' \
        'setuptools<81' einops tqdm pillow numpy huggingface_hub iopath hydra-core \
        timm pycocotools omegaconf psutil \
        || echo "⚠️  No se pudo instalar sam3 — instálalo a mano dentro de $ENV_SAM3."
    echo "ℹ️  El checkpoint facebook/sam3 es un repo gated en Hugging Face:"
    echo "    solicita acceso en https://huggingface.co/facebook/sam3 y autentícate con"
    echo "    $ENV_SAM3/bin/huggingface-cli login   (o exporta HF_TOKEN)"
else
    echo "══ [3/3] Venv de SAM3 omitido (usa --sam3 para crearlo) ══"
fi

echo ""
echo "✅ Instalación terminada."
echo "   Activa el entorno:   source venv/bin/activate"
echo "   CLI multi-agente:    python main.py --help"
echo "   Dashboard:           streamlit run src/dashboard/app.py"
