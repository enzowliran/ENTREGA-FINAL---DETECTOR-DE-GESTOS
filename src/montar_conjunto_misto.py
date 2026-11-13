"""Teste curto com todas as bases juntas: monta as amostras de IPN, DriverMHG, HaGRID e Slovo para os dois caminhos.

Cada amostra sai com:
  - os quadros para a CNN 3D: até 36 quadros a 12 por segundo, quadrado central em 128x128, em JPG;
  - os pontos da mão para a GRU: MediaPipe (modo vídeo, um landmarker novo por amostra, confiança 0,3) nos
    quadros inteiros, nos mesmos instantes. Fotos do HaGRID viram 36 quadros iguais.
O Jester não entra aqui: a CNN lê do próprio arquivo e os pontos já estão em data/pontos/jester_amostra.pkl.

Classes unificadas: as 27 do Jester, um a quatro dedos e os 5 gestos do DriverMHG ("Volante: ..."). "Nenhum gesto"
e "outra coisa" de todas as bases vão para "No gesture" e "Doing other things".

Uso:
    demo\\python_pontos.bat src\\montar_conjunto_misto.py
"""
import argparse
import collections
import csv
import json
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pontos_mao as P

RAIZ = Path(__file__).resolve().parent.parent
QPS, MAXQ, LADO = 12, 36, 128
IPN = {"B0A": "One Finger", "B0B": "Two Fingers", "D0X": "No gesture", "G01": "Doing other things", "G02": "Doing other things",
       "G07": "Doing other things", "G08": "Doing other things", "G09": "Doing other things"}
DRIVER = {1: "Volante: FlickDown", 2: "Volante: FlickUp", 3: "Volante: SwipeLeft", 4: "Volante: SwipeRight", 5: "Volante: Tap"}
HAGRID = {"one": "One Finger", "peace": "Two Fingers", "two_up": "Two Fingers", "three": "Three Fingers", "three2": "Three Fingers",
          "four": "Four Fingers"}


def novo_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    o = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(P.baixar_modelo_mao())),
                                     running_mode=vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=0.3,
                                     min_hand_presence_confidence=0.3, min_tracking_confidence=0.5)
    return mp, vision.HandLandmarker.create_from_options(o)


def quadrado(bgr, lado=LADO):
    a, l = bgr.shape[:2]
    b = min(a, l)
    q = bgr[(a - b) // 2:(a - b) // 2 + b, (l - b) // 2:(l - b) // 2 + b]
    return cv2.imencode(".jpg", cv2.resize(q, (lado, lado), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


class Sequencia:
    """Junta os quadros de uma amostra: JPG para a CNN e pontos do MediaPipe (quadro inteiro, até 640 de largura)."""

    def __init__(self):
        self.mp, self.lm = novo_landmarker()
        self.jpgs, self.img, self.world, self.present, self.score, self.al = [], [], [], [], [], None

    def adicionar(self, bgr, jpg=None):
        self.jpgs.append(jpg if jpg is not None else quadrado(bgr))
        a, l = bgr.shape[:2]
        if l > 640:
            bgr = cv2.resize(bgr, (640, round(a * 640 / l)), interpolation=cv2.INTER_AREA)
        self.al = bgr.shape[:2]
        r = self.lm.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(bgr[..., ::-1])),
                                     int(round(len(self.jpgs) * 1000 / QPS)))
        img, world = np.zeros((2, 21, 3), np.float32), np.zeros((2, 21, 3), np.float32)
        pres, sc = np.zeros(2, bool), np.zeros(2, np.float32)
        for s, k in enumerate(sorted(range(len(r.hand_landmarks)), key=lambda k: -r.handedness[k][0].score)[:2]):
            img[s] = [(p.x, p.y, p.z) for p in r.hand_landmarks[k]]
            world[s] = [(p.x, p.y, p.z) for p in r.hand_world_landmarks[k]]
            pres[s], sc[s] = True, r.handedness[k][0].score
        self.img.append(img), self.world.append(world), self.present.append(pres), self.score.append(sc)

    def fechar(self, **info):
        self.lm.close()
        return {**info, "jpgs": self.jpgs, "img": np.stack(self.img), "world": np.stack(self.world), "present": np.stack(self.present),
                "score": np.stack(self.score), "altura_largura": self.al}


def indices_12(ini, fim, fps):
    """Índices (base 0) a 12 por segundo entre ini e fim (inclusive), no máximo 3 s centrados."""
    idx = np.arange(ini, fim + 1, fps / QPS).astype(int)
    if len(idx) > MAXQ:
        c = (len(idx) - MAXQ) // 2
        idx = idx[c:c + MAXQ]
    return idx


def tarefa_video(args):
    """Um vídeo (IPN ou Slovo) lido em sequência; cada amostra é um trecho [(id, info, índices)] sem sobreposição."""
    caminho, amostras = args
    cap = cv2.VideoCapture(str(caminho))
    alvo = {}
    for k, (_, _, idx) in enumerate(amostras):
        for i in idx:
            alvo.setdefault(int(i), []).append(k)
    seqs, saida, i = {}, [], 0
    ultimo = {k: int(idx[-1]) for k, (_, _, idx) in enumerate(amostras)}
    fim_total = max(alvo) if alvo else -1
    while i <= fim_total:
        ok, q = cap.read()
        if not ok:
            break
        for k in alvo.get(i, []):
            seqs.setdefault(k, Sequencia()).adicionar(q)
            if i == ultimo[k]:
                saida.append(seqs.pop(k).fechar(**amostras[k][1]))
        i += 1
    for k, s in seqs.items():
        saida.append(s.fechar(**amostras[k][1]))
    cap.release()
    return saida


def tarefa_driver(args):
    nome, amostras = args
    rgb = np.load(RAIZ / f"data/DriverMHG/prep/{nome}_rgb.npy", mmap_mode="r")
    saida = []
    for info, idx in amostras:
        s = Sequencia()
        for i in idx:
            if i < len(rgb):
                s.adicionar(np.ascontiguousarray(rgb[i][..., ::-1]))  # o npy é RGB
        if s.jpgs:
            saida.append(s.fechar(**info))
        else:
            s.lm.close()
    return saida


def tarefa_hagrid(lote):
    saida = []
    for arquivo, caixa, info in lote:
        img = cv2.imread(str(arquivo))
        if img is None:
            continue
        a, l = img.shape[:2]
        x, y, w, h = caixa
        cx, cy, lado = (x + w / 2) * l, (y + h / 2) * a, max(w * l, h * a) * 2.5
        x0, y0 = int(max(cx - lado / 2, 0)), int(max(cy - lado / 2, 0))
        rec = img[y0:int(min(cy + lado / 2, a)), x0:int(min(cx + lado / 2, l))]
        if rec.size == 0:
            continue
        jpg = quadrado(rec)
        s = Sequencia()
        for _ in range(MAXQ):  # foto parada repetida: o rastreamento do modo vídeo trabalha como numa mão parada
            s.adicionar(rec, jpg)
        saida.append(s.fechar(**info))
    return saida


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--treino", type=int, default=200, help="amostras de treino por classe e base")
    a.add_argument("--teste", type=int, default=100, help="amostras de teste por classe e base")
    a.add_argument("--processos", type=int, default=7)
    args = a.parse_args()
    g = np.random.default_rng(0)
    limite = {"treino": args.treino, "teste": args.teste}

    def sortear(itens, chave):
        por = collections.defaultdict(list)
        for it in itens:
            por[chave(it)].append(it)
        saida = []
        for (div, _), lista in por.items():
            ordem = g.permutation(len(lista))[:limite[div]]
            saida += [lista[i] for i in ordem]
        return saida

    # IPN: trechos anotados; "sem gesto" cortado em janelas de 3 s
    meta = {l["Video Name"]: l["Set"] for l in csv.DictReader(open(RAIZ / "data/IPN/anotacoes/metadata.csv", encoding="utf-8"))}
    itens = []
    for l in csv.DictReader(open(RAIZ / "data/IPN/anotacoes/Annot_List.txt", encoding="utf-8")):
        if l["label"] not in IPN:
            continue
        div = "treino" if meta[l["video"]] == "train" else "teste"
        ini, fim = int(l["t_start"]) - 1, int(l["t_end"]) - 1
        blocos = [(ini, fim)] if l["label"] != "D0X" else [(s, min(s + 89, fim)) for s in range(ini, fim - 29, 90)]
        for b0, b1 in blocos:
            itens.append({"fonte": "IPN", "classe": IPN[l["label"]], "divisao": div, "video": l["video"], "pessoa": "_".join(l["video"].split("_")[:2]),
                          "idx": indices_12(b0, b1, 30.0)})
    ipn = sortear([i for i in itens if len(i["idx"]) >= 6], lambda i: (i["divisao"], i["classe"]))
    tarefas = collections.defaultdict(list)
    for it in ipn:
        tarefas[RAIZ / f"data/IPN/videos/videos/{it['video']}.avi"].append((None, {k: v for k, v in it.items() if k != "idx"}, it["idx"]))

    # Slovo: só "sem sinal"
    itens = []
    for l in csv.DictReader(open(RAIZ / "data/Slovo/annotations.tsv", encoding="utf-8"), delimiter="\t"):
        c = RAIZ / f"data/Slovo/videos/{l['attachment_id']}.mp4"
        if l["text"] == "no_event" and c.exists():
            itens.append({"fonte": "Slovo", "classe": "No gesture", "divisao": "treino" if l["train"] == "True" else "teste",
                          "video": l["attachment_id"], "caminho": c})
    for it in sortear(itens, lambda i: (i["divisao"], i["classe"])):
        cap = cv2.VideoCapture(str(it["caminho"]))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        tarefas[it.pop("caminho")].append((None, it, indices_12(0, max(n - 1, 0), fps)))

    # DriverMHG: 4 pessoas no disco (1, 2, 10, 11); a 11 fica para o teste
    driver = collections.defaultdict(list)
    itens = []
    for arq in sorted((RAIZ / "data/DriverMHG/prep").glob("*_anotacao.txt")):
        nome = arq.name.replace("_anotacao.txt", "")
        pessoa = int(re.match(r"S(\d+)", nome)[1])
        for linha in arq.read_text().splitlines():
            p = linha.split()
            if len(p) != 4:
                continue
            c, ini, fim = int(p[1]), int(p[2]) - 1, int(p[3]) - 1
            classe = DRIVER.get(c if c <= 5 else c - 5) if c <= 10 else ("No gesture" if c <= 12 else "Doing other things")
            itens.append({"fonte": "DriverMHG", "classe": classe, "divisao": "teste" if pessoa == 11 else "treino", "video": nome,
                          "pessoa": pessoa, "idx": indices_12(ini, fim, 30.0)})
    for it in sortear([i for i in itens if len(i["idx"]) >= 4], lambda i: (i["divisao"], i["classe"])):
        driver[it["video"]].append(({k: v for k, v in it.items() if k != "idx"}, it.pop("idx")))

    # HaGRID: números (fotos)
    anot = json.load(open(RAIZ / "data/HaGRID/anotacoes.json", encoding="utf-8"))
    itens = []
    for pasta, div in (("train", "treino"), ("val", "teste")):
        for classe_dir in sorted((RAIZ / "data/HaGRID" / pasta).iterdir()):
            if classe_dir.name not in HAGRID:
                continue
            for f in sorted(classe_dir.glob("*.jpg")):
                info = anot.get(f.stem)
                if info:
                    k = info["labels"].index(classe_dir.name) if classe_dir.name in info["labels"] else 0
                    itens.append((f, info["bboxes"][k], {"fonte": "HaGRID", "classe": HAGRID[classe_dir.name], "divisao": div,
                                                        "video": f.stem, "pessoa": info["user_id"]}))
    hagrid = sortear(itens, lambda i: (i[2]["divisao"], i[2]["classe"]))
    del anot

    print(f"IPN {len(ipn)}, Slovo {sum(1 for v in tarefas.values() for x in v if x[1]['fonte'] == 'Slovo')}, "
          f"DriverMHG {sum(len(v) for v in driver.values())}, HaGRID {len(hagrid)} amostras", flush=True)
    t0, saida = time.time(), []
    with ProcessPoolExecutor(args.processos) as ex:
        for r in ex.map(tarefa_video, [(c, sorted(v, key=lambda x: x[2][0])) for c, v in tarefas.items()]):
            saida += r
        print(f"  IPN e Slovo prontos ({time.time() - t0:.0f} s)", flush=True)
        for r in ex.map(tarefa_driver, list(driver.items())):
            saida += r
        print(f"  DriverMHG pronto ({time.time() - t0:.0f} s)", flush=True)
        for r in ex.map(tarefa_hagrid, [hagrid[i:i + 40] for i in range(0, len(hagrid), 40)]):
            saida += r
    pickle.dump(saida, open(RAIZ / "data/pontos/misto.pkl", "wb"))
    resumo = collections.Counter((s["fonte"], s["divisao"], s["classe"]) for s in saida)
    for k in sorted(resumo):
        print(f"  {k[0]} · {k[1]} · {k[2]}: {resumo[k]}  (mão em {100 * np.mean([s['present'][:, 0].mean() for s in saida if (s['fonte'], s['divisao'], s['classe']) == k]):.0f}% dos quadros)")
    print(f"Pronto em {(time.time() - t0) / 60:.1f} min: {len(saida)} amostras")


if __name__ == "__main__":
    main()
