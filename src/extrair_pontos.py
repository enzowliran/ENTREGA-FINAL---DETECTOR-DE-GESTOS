"""Extrai uma vez os pontos da mão (MediaPipe Hand Landmarker) e guarda em cache, para testar regras e redes sem rodar o MediaPipe de novo.

Saída em data/pontos/ (fora do git; o ASL Citizen não pode ser redistribuído):
  - ipn.pkl: os 200 vídeos do IPN Hand, a `--qps` quadros por segundo, com os trechos anotados e a divisão (train/test);
  - curtos.pkl: Slovo (no_event, "um", "dois") e ASL Citizen ("ONE", "TWO");
  - hagrid.pkl: fotos do HaGRID recortadas em volta da mão do gesto (a mesma regra do avaliar_pontos_hagrid.py).

Cada quadro guarda: tempo, mão achada, 21 pontos na imagem, 21 pontos em metros, confiança e lado.

Uso:
    demo\\python_pontos.bat src\\extrair_pontos.py
"""
import argparse
import csv
import json
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
DESTINO = RAIZ / "data/pontos"


def quadro_vazio(t):
    return {"t": t, "mao": False}


def mao_dict(t, mao):
    if mao is None:
        return quadro_vazio(t)
    return {"t": t, "mao": True, "imagem": mao.imagem.astype(np.float32), "mundo": mao.mundo.astype(np.float32),
            "conf": mao.confianca, "lado": mao.lado}


def video(args):
    caminho, qps = args
    detector = P.DetectorMao(video=True)
    cap = cv2.VideoCapture(str(caminho))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    quadros, i, proximo, tam = [], 0, 0.0, None
    while True:
        ok, q = cap.read()
        if not ok:
            break
        if i >= proximo:
            tam = q.shape[:2]
            quadros.append(mao_dict(i / fps, detector.detectar(q, agora=i / fps)))
            proximo += fps / qps
        i += 1
    cap.release()
    detector.fechar()
    return {"fps": fps, "duracao": i / fps, "altura_largura": tam, "quadros": quadros}


def fotos(lote):
    detector = P.DetectorMao(video=False)
    saida = []
    for caminho, caixa, classe, divisao, pessoa in lote:
        img = cv2.imread(str(caminho))
        if img is None:
            continue
        a, l = img.shape[:2]
        x, y, w, h = caixa
        cx, cy, lado = (x + w / 2) * l, (y + h / 2) * a, max(w * l, h * a) * 2.0
        x0, y0 = int(max(cx - lado / 2, 0)), int(max(cy - lado / 2, 0))
        recorte = img[y0:int(min(cy + lado / 2, a)), x0:int(min(cx + lado / 2, l))]
        if recorte.size == 0:
            continue
        r = mao_dict(0.0, detector.detectar(recorte))
        r.update(classe=classe, divisao=divisao, pessoa=pessoa)
        saida.append(r)
    detector.fechar()
    return saida


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--qps", type=float, default=15.0)
    a.add_argument("--processos", type=int, default=6)
    a.add_argument("--hagrid-treino", type=int, default=1500, help="fotos de treino por classe")
    a.add_argument("--hagrid-val", type=int, default=400, help="fotos de validação por classe")
    a.add_argument("--so", choices=("ipn", "curtos", "hagrid"), nargs="*")
    args = a.parse_args()
    DESTINO.mkdir(parents=True, exist_ok=True)
    fazer = set(args.so or ("ipn", "curtos", "hagrid"))
    t0 = time.time()

    with ProcessPoolExecutor(args.processos) as ex:
        if "ipn" in fazer:
            meta = {l["Video Name"]: l["Set"] for l in csv.DictReader(open(RAIZ / "data/IPN/anotacoes/metadata.csv", encoding="utf-8"))}
            anot = {}
            for l in csv.DictReader(open(RAIZ / "data/IPN/anotacoes/Annot_List.txt", encoding="utf-8")):
                anot.setdefault(l["video"], []).append((int(l["t_start"]), int(l["t_end"]), l["label"]))
            nomes = sorted(anot)
            ipn = {}
            for k, (v, r) in enumerate(zip(nomes, ex.map(video, [(RAIZ / f"data/IPN/videos/videos/{v}.avi", args.qps) for v in nomes])), 1):
                r.update(trechos=anot[v], divisao=meta[v], pessoa="_".join(v.split("_")[:2]))
                ipn[v] = r
                if k % 20 == 0:
                    print(f"  IPN {k}/{len(nomes)} ({time.time() - t0:.0f} s)", flush=True)
            pickle.dump(ipn, open(DESTINO / "ipn.pkl", "wb"))

        if "curtos" in fazer:
            itens = []
            for l in csv.DictReader(open(RAIZ / "data/Slovo/annotations.tsv", encoding="utf-8"), delimiter="\t"):
                c = RAIZ / f"data/Slovo/videos/{l['attachment_id']}.mp4"
                if not c.exists():
                    continue
                if l["text"] == "no_event":
                    itens.append({"base": "Slovo sem sinal", "caminho": c, "classe": None, "divisao": "train" if l["train"] == "True" else "test",
                                  "pessoa": l["user_id"], "inicio": None, "fim": None})
                elif l["text"] in ("один", "два"):
                    itens.append({"base": "Slovo sinais", "caminho": c, "classe": "One Finger" if l["text"] == "один" else "Two Fingers",
                                  "divisao": "test", "pessoa": l["user_id"], "inicio": int(l["begin"]), "fim": int(l["end"])})
            for s in ("train", "val", "test"):
                for l in csv.DictReader(open(RAIZ / f"data/ASLCitizen/{s}.csv", encoding="utf-8")):
                    c = RAIZ / "data/ASLCitizen/videos" / Path(l["Video file"]).name
                    if l["Gloss"] in ("ONE", "TWO") and c.exists():
                        itens.append({"base": "ASL Citizen sinais", "caminho": c, "classe": "One Finger" if l["Gloss"] == "ONE" else "Two Fingers",
                                      "divisao": "test", "pessoa": l["Participant ID"], "inicio": None, "fim": None})
            for it, r in zip(itens, ex.map(video, [(it["caminho"], args.qps) for it in itens], chunksize=4)):
                it.update(r)
                it["caminho"] = str(it["caminho"].relative_to(RAIZ))
            pickle.dump(itens, open(DESTINO / "curtos.pkl", "wb"))
            print(f"  curtos: {len(itens)} vídeos ({time.time() - t0:.0f} s)", flush=True)

        if "hagrid" in fazer:
            anot = json.load(open(RAIZ / "data/HaGRID/anotacoes.json", encoding="utf-8"))
            gerador = np.random.default_rng(0)
            lotes = []
            for divisao, pasta, n in (("train", "train", args.hagrid_treino), ("test", "val", args.hagrid_val)):
                for classe_dir in sorted((RAIZ / "data/HaGRID" / pasta).iterdir()):
                    arquivos = sorted(classe_dir.glob("*.jpg"))
                    arquivos = [arquivos[i] for i in gerador.permutation(len(arquivos))[:n]]
                    itens = []
                    for f in arquivos:
                        info = anot.get(f.stem)
                        if not info:
                            continue
                        rotulos = info["labels"]
                        alvo = classe_dir.name
                        k = rotulos.index(alvo) if alvo in rotulos else 0
                        itens.append((f, info["bboxes"][k], alvo, divisao, info["user_id"]))
                    lotes += [itens[i:i + 100] for i in range(0, len(itens), 100)]
            hagrid = [r for lote in ex.map(fotos, lotes) for r in lote]
            pickle.dump(hagrid, open(DESTINO / "hagrid.pkl", "wb"))
            print(f"  HaGRID: {len(hagrid)} fotos, mão em {100 * np.mean([h['mao'] for h in hagrid]):.1f}% ({time.time() - t0:.0f} s)", flush=True)
    print(f"Pronto em {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
