"""Teste pequeno: extrai os pontos da mão (MediaPipe) de uma amostra do Jester, para treinar uma rede sobre coordenadas.

Configuração escolhida no piloto (resultados/pontos/piloto_jester.json): tamanho original, modo vídeo,
confiança 0,3, rastreamento 0,5, até 2 mãos, um landmarker novo por clipe.

Saída: data/pontos/jester_amostra.pkl (fora do git), com, por clipe: classe, divisão, quadros com
img [T,2,21,3], world [T,2,21,3], present [T,2], handedness [T,2] (rótulo cru do MediaPipe) e score [T,2].

Uso:
    demo\\python_pontos.bat src\\extrair_pontos_jester.py --treino 300 --validacao 100
"""
import argparse
import collections
import pickle
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
_MAPA = None


def _iniciar():
    global _MAPA
    _MAPA = np.memmap(RAIZ / "data/Jester/jester_frames.bin", dtype=np.uint8, mode="r")


def extrair(linhas):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opcoes = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(P.baixar_modelo_mao())), running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.5)
    T = len(linhas)
    img = np.zeros((T, 2, 21, 3), np.float32)
    world = np.zeros((T, 2, 21, 3), np.float32)
    present = np.zeros((T, 2), bool)
    handed = np.full((T, 2), -1, np.int8)
    score = np.zeros((T, 2), np.float32)
    tamanho = None
    with vision.HandLandmarker.create_from_options(opcoes) as lm:
        for t, (d, n) in enumerate(linhas):
            bgr = cv2.imdecode(np.asarray(_MAPA[d:d + n]), cv2.IMREAD_COLOR)
            tamanho = bgr.shape[:2]
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(bgr[..., ::-1])),
                                    int(round(t * 1000 / 12)))
            # a mão mais confiante fica no slot 0
            ordem = sorted(range(len(r.hand_landmarks)), key=lambda k: -r.handedness[k][0].score)[:2]
            for s, k in enumerate(ordem):
                img[t, s] = [(p.x, p.y, p.z) for p in r.hand_landmarks[k]]
                world[t, s] = [(p.x, p.y, p.z) for p in r.hand_world_landmarks[k]]
                handed[t, s] = 0 if r.handedness[k][0].category_name == "Left" else 1
                score[t, s] = r.handedness[k][0].score
                present[t, s] = True
    return {"img": img, "world": world, "present": present, "handedness": handed, "score": score, "altura_largura": tamanho}


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--treino", type=int, default=300, help="clipes de treino por classe")
    a.add_argument("--validacao", type=int, default=100, help="clipes da validação oficial por classe")
    a.add_argument("--processos", type=int, default=7)
    args = a.parse_args()
    rot = RAIZ / "data/Jester/rotulos/jester_labels"
    nomes = [l.strip() for l in (rot / "labels.csv").read_text().splitlines() if l.strip()]
    g = np.random.default_rng(1)
    clipes = []
    for arquivo, divisao, n in (("train.csv", "treino", args.treino), ("validation.csv", "validacao", args.validacao)):
        por = collections.defaultdict(list)
        for linha in (rot / arquivo).read_text().splitlines():
            if ";" in linha:
                v, c = linha.split(";", 1)
                por[c.strip()].append(int(v))
        clipes += [(int(v), c, divisao) for c in nomes for v in g.choice(por[c], n, replace=False)]
    indice = np.fromfile(RAIZ / "data/Jester/jester_indice.bin", dtype=TIPO_INDICE)
    indice = indice[np.isin(indice["video"], [v for v, _, _ in clipes])]
    indice = indice[np.lexsort((indice["frame"], indice["video"]))]
    linhas = collections.defaultdict(list)
    for r in indice:
        linhas[int(r["video"])].append((int(r["deslocamento"]), int(r["tamanho"])))
    print(f"{len(clipes)} clipes", flush=True)
    t0, saida = time.time(), []
    with ProcessPoolExecutor(args.processos, initializer=_iniciar) as ex:
        for k, ((v, c, d), r) in enumerate(zip(clipes, ex.map(extrair, [linhas[v] for v, _, _ in clipes], chunksize=8)), 1):
            r.update(video=v, classe=c, divisao=d)
            saida.append(r)
            if k % 1000 == 0:
                print(f"  {k}/{len(clipes)} ({time.time() - t0:.0f} s)", flush=True)
    pickle.dump({"nomes": nomes, "clipes": saida}, open(RAIZ / "data/pontos/jester_amostra.pkl", "wb"))
    print(f"Pronto em {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
