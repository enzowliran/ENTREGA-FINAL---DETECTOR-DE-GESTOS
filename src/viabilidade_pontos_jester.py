"""Passo 1 da ideia da LSTM sobre os pontos (diário de 12/11): o Hand Landmarker acha a mão nos vídeos do Jester?

Sorteia `--por-classe` vídeos de validação de cada uma das 27 classes, passa todos os quadros pelo
Hand Landmarker em modo vídeo (12 quadros por segundo, como o Jester) e mede em quantos quadros ele acha a mão.
Critério do plano: pelo menos 80% dos quadros nos gestos (as classes sem gesto ficam à parte).

Uso (precisa só do MediaPipe):
    demo\\python_pontos.bat src\\viabilidade_pontos_jester.py
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pontos_mao as P

RAIZ = Path(__file__).resolve().parent.parent
TIPO_INDICE = np.dtype([("video", "<i4"), ("frame", "<i2"), ("deslocamento", "<i8"), ("tamanho", "<i4")])
SEM_GESTO = ("No gesture", "Doing other things")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--por-classe", type=int, default=8)
    a.add_argument("--ampliar", type=float, default=1.0, help="ampliar os quadros antes do MediaPipe (o Jester tem ~100 px de altura)")
    a.add_argument("--saida", default="resultados/pontos/viabilidade_jester.json")
    args = a.parse_args()

    pasta = RAIZ / "data/Jester"
    rot = pasta / "rotulos/jester_labels"
    nomes = [l.strip() for l in (rot / "labels.csv").read_text().splitlines() if l.strip()]
    por_classe = collections.defaultdict(list)
    for linha in (rot / "validation.csv").read_text().splitlines():
        if ";" in linha:
            v, n = linha.split(";", 1)
            por_classe[n.strip()].append(int(v))
    gerador = np.random.default_rng(0)
    escolhidos = {n: set(gerador.choice(por_classe[n], args.por_classe, replace=False).tolist()) for n in nomes}
    todos = set().union(*escolhidos.values())

    indice = np.fromfile(pasta / "jester_indice.bin", dtype=TIPO_INDICE)
    indice = indice[np.isin(indice["video"], list(todos))]
    indice = indice[np.lexsort((indice["frame"], indice["video"]))]
    mapa = np.memmap(pasta / "jester_frames.bin", dtype=np.uint8, mode="r")

    resultado, alturas = {}, []
    for n in nomes:
        quadros = achados = 0
        for v in sorted(escolhidos[n]):
            detector = P.DetectorMao(video=True, largura=0)
            for k, r in enumerate(indice[indice["video"] == v]):
                img = cv2.imdecode(np.asarray(mapa[r["deslocamento"]:r["deslocamento"] + r["tamanho"]]), cv2.IMREAD_COLOR)
                alturas.append(img.shape[0])
                if args.ampliar != 1:
                    img = cv2.resize(img, None, fx=args.ampliar, fy=args.ampliar, interpolation=cv2.INTER_CUBIC)
                quadros += 1
                achados += detector.detectar(img, agora=k / 12) is not None
            detector.fechar()
        resultado[n] = round(100 * achados / max(quadros, 1), 1)
        print(f"{n}: mão em {resultado[n]}% dos {quadros} quadros", flush=True)

    gestos = [resultado[n] for n in nomes if n not in SEM_GESTO]
    saida = {"videos_por_classe": args.por_classe, "ampliar": args.ampliar, "altura_mediana_px": int(np.median(alturas)),
             "media_gestos_pct": round(float(np.mean(gestos)), 1), "pior_gesto_pct": min(gestos),
             "passa_criterio_80": bool(np.mean(gestos) >= 80), "por_classe_pct": resultado}
    destino = RAIZ / args.saida
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in saida.items() if k != "por_classe_pct"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
