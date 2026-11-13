"""Etapa 2 do plano de coordenadas: piloto do MediaPipe no Jester (portão de decisão).

Para cada configuração (escala x modo x confiança de detecção/presença; rastreamento 0,5) e cada classe:
  - % de quadros com pelo menos uma mão;
  - % de clipes com mão em >= 70% dos quadros;
  - tempo por quadro.
Portão: nas 25 classes com mão (sem "No gesture" e "Doing other things"), a mediana de
"% de clipes com mão em >= 70% dos quadros" >= 85% e nenhuma classe abaixo de 60%.

Amostra: `--por-classe` clipes de TREINO por classe (semente 0). Um landmarker novo por clipe.

Uso:
    demo\\python_pontos.bat src\\piloto_jester_mediapipe.py
"""
import argparse
import collections
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pontos_mao as P

RAIZ = Path(__file__).resolve().parent.parent
TIPO_INDICE = np.dtype([("video", "<i4"), ("frame", "<i2"), ("deslocamento", "<i8"), ("tamanho", "<i4")])
SEM_MAO = ("No gesture", "Doing other things")
_IDX = _MAPA = None


def _iniciar():
    global _MAPA
    _MAPA = np.memmap(RAIZ / "data/Jester/jester_frames.bin", dtype=np.uint8, mode="r")


def rodar(args):
    """Um clipe numa configuração: devolve (quadros, quadros com mão, segundos)."""
    linhas, escala, modo, conf = args
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opcoes = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(P.baixar_modelo_mao())),
        running_mode=vision.RunningMode.VIDEO if modo == "video" else vision.RunningMode.IMAGE,
        num_hands=2, min_hand_detection_confidence=conf, min_hand_presence_confidence=conf, min_tracking_confidence=0.5)
    achados, t0 = 0, time.perf_counter()
    with vision.HandLandmarker.create_from_options(opcoes) as lm:
        for k, (d, t) in enumerate(linhas):
            bgr = cv2.imdecode(np.asarray(_MAPA[d:d + t]), cv2.IMREAD_COLOR)
            if escala != 1:
                bgr = cv2.resize(bgr, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(bgr[..., ::-1]))
            r = lm.detect_for_video(img, int(round(k * 1000 / 12))) if modo == "video" else lm.detect(img)
            achados += bool(r.hand_landmarks)
    return len(linhas), achados, time.perf_counter() - t0


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--por-classe", type=int, default=20)
    a.add_argument("--processos", type=int, default=7)
    a.add_argument("--saida", default="resultados/pontos/piloto_jester.json")
    args = a.parse_args()
    rot = RAIZ / "data/Jester/rotulos/jester_labels"
    nomes = [l.strip() for l in (rot / "labels.csv").read_text().splitlines() if l.strip()]
    por_classe = collections.defaultdict(list)
    for linha in (rot / "train.csv").read_text().splitlines():
        if ";" in linha:
            v, n = linha.split(";", 1)
            por_classe[n.strip()].append(int(v))
    g = np.random.default_rng(0)
    clipes = [(n, int(v)) for n in nomes for v in g.choice(por_classe[n], args.por_classe, replace=False)]
    indice = np.fromfile(RAIZ / "data/Jester/jester_indice.bin", dtype=TIPO_INDICE)
    indice = indice[np.isin(indice["video"], [v for _, v in clipes])]
    indice = indice[np.lexsort((indice["frame"], indice["video"]))]
    linhas = {v: [] for _, v in clipes}
    for r in indice:
        linhas[int(r["video"])].append((int(r["deslocamento"]), int(r["tamanho"])))

    configs = list(itertools.product((1, 2, 3, 4), ("video", "image"), (0.3, 0.5)))
    resultado = {"clipes_por_classe": args.por_classe, "clipes": len(clipes), "configuracoes": []}
    t0 = time.time()
    with ProcessPoolExecutor(args.processos, initializer=_iniciar) as ex:
        for escala, modo, conf in configs:
            saidas = list(ex.map(rodar, [(linhas[v], escala, modo, conf) for _, v in clipes], chunksize=8))
            classe = {}
            for n in nomes:
                s = [o for (c, _), o in zip(clipes, saidas) if c == n]
                classe[n] = {"pct_quadros_mao": round(100 * sum(x[1] for x in s) / max(sum(x[0] for x in s), 1), 1),
                             "pct_clipes_70": round(100 * np.mean([x[1] >= 0.7 * x[0] for x in s]), 1)}
            com_mao = [classe[n]["pct_clipes_70"] for n in nomes if n not in SEM_MAO]
            ms = 1000 * sum(x[2] for x in saidas) / max(sum(x[0] for x in saidas), 1)
            c = {"escala": escala, "modo": modo, "det_pres": conf, "ms_por_quadro": round(ms, 1),
                 "mediana_pct_clipes_70": float(np.median(com_mao)), "pior_classe_pct_clipes_70": min(com_mao),
                 "pct_quadros_mao_25_classes": round(float(np.mean([classe[n]["pct_quadros_mao"] for n in nomes if n not in SEM_MAO])), 1),
                 "passa_portao": bool(np.median(com_mao) >= 85 and min(com_mao) >= 60), "por_classe": classe}
            resultado["configuracoes"].append(c)
            print(f"escala {escala} {modo:5s} conf {conf}: quadros com mão {c['pct_quadros_mao_25_classes']}%, "
                  f"mediana clipes>=70% {c['mediana_pct_clipes_70']}%, pior {c['pior_classe_pct_clipes_70']}%, "
                  f"{c['ms_por_quadro']} ms/quadro ({time.time() - t0:.0f} s)", flush=True)
            destino = RAIZ / args.saida
            destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
