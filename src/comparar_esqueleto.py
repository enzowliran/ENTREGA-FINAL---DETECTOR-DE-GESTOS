"""Quarto modelo do teste misto: MediaPipe primeiro, depois a CNN 3D sobre o esqueleto da mão desenhado.

Mesmas amostras e mesmo critério de src/comparar_misto.py. Em cada um dos 16 quadros do clipe, os 21 pontos de cada
mão viram um desenho em 64x64 (linhas dos ossos e pontos das articulações), num recorte quadrado em volta de onde a
mão andou no clipe todo (como no PoseC3D). Sem mão, o quadro fica preto. A CNN 3D (cnn_3d, 1 canal) treina do zero.

Uso:
    demo\\python.bat src\\comparar_esqueleto.py
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import jester as J
import modelos as M
import pontos_mao as P
from comparar_misto import EXTRAS

RAIZ = Path(__file__).resolve().parent.parent
N, LADO = 16, 64


class ConjuntoEsqueleto:
    def __init__(self, amostras, treino):
        self.amostras, self.treino = amostras, treino

    def __len__(self):
        return len(self.amostras)

    def carregar(self, posicao, g):
        s = self.amostras[posicao]
        T = len(s["present"])
        pontos = np.linspace(0, T, N + 1)
        if self.treino:
            idx = (pontos[:-1] + g.random(N) * (pontos[1:] - pontos[:-1])).astype(int)
        else:
            idx = ((pontos[:-1] + pontos[1:]) / 2).astype(int)
        idx = np.minimum(idx, T - 1)
        a, l = s["altura_largura"] or (100, 176)
        xy = s["img"][..., :2] * np.array([l, a], np.float32)  # [T, 2, 21, 2] em pixels
        pres = s["present"]
        clipe = np.zeros((N, 1, LADO, LADO), np.float32)
        if pres.any():
            todos = xy[pres].reshape(-1, 2)
            lo, hi = todos.min(0), todos.max(0)
            centro, lado = (lo + hi) / 2, max((hi - lo).max() * 1.2, 1e-3)
            ang, esc, desl = 0.0, 1.0, np.zeros(2)
            if self.treino:
                ang, esc, desl = np.deg2rad(g.uniform(-10, 10)), g.uniform(0.9, 1.1), g.uniform(-0.05, 0.05, 2) * lado
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * esc
            for k, t in enumerate(idx):
                quadro = np.zeros((LADO, LADO), np.uint8)
                for m in range(2):
                    if not pres[t, m]:
                        continue
                    p = ((xy[t, m] - centro - desl) @ R.T / lado + 0.5) * (LADO - 1)
                    p = np.round(p).astype(int)
                    for i, j in P.LIGACOES:
                        cv2.line(quadro, tuple(p[i]), tuple(p[j]), 160, 1, cv2.LINE_AA)
                    for i in range(21):
                        cv2.circle(quadro, tuple(p[i]), 1, 255, -1)
                clipe[k, 0] = quadro
        return clipe, s["y"]


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--epocas", type=int, default=15)
    a.add_argument("--saida", default="resultados/pontos/misto_esqueleto.json")
    args = a.parse_args()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    jd = pickle.load(open(RAIZ / "data/pontos/jester_amostra.pkl", "rb"))
    nomes = list(jd["nomes"]) + EXTRAS
    idx = {n: i for i, n in enumerate(nomes)}
    amostras = [{"fonte": "Jester", "y": idx[c["classe"]], "divisao": "treino" if c["divisao"] == "treino" else "teste",
                 **{k: c[k] for k in ("img", "present", "altura_largura")}} for c in jd["clipes"]]
    del jd
    for s in pickle.load(open(RAIZ / "data/pontos/misto.pkl", "rb")):
        amostras.append({"fonte": s["fonte"], "y": idx[s["classe"]], "divisao": s["divisao"],
                         **{k: s[k] for k in ("img", "present", "altura_largura")}})
    tr = [s for s in amostras if s["divisao"] == "treino"]
    te = [s for s in amostras if s["divisao"] == "teste"]
    c_tr, c_te = ConjuntoEsqueleto(tr, True), ConjuntoEsqueleto(te, False)
    print(f"treino {len(tr)}, teste {len(te)}", flush=True)

    m = M.criar_modelo("cnn_3d", len(nomes), canais_entrada=1).cuda()
    otim = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    passos = len(c_tr) // 48 + 1
    agenda = torch.optim.lr_scheduler.OneCycleLR(otim, 1e-3, total_steps=args.epocas * passos, pct_start=0.1)
    preparar = lambda x: (x.cuda(non_blocking=True) / 255.0 - 0.1) / 0.3

    @torch.no_grad()
    def prever():
        m.eval()
        p = []
        for x, _ in J.lotes(c_te, 128, seed=0, embaralhar=False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p.append(m(preparar(x)).float().argmax(1).cpu())
        m.train()
        return torch.cat(p).numpy()

    Yt = np.array([s["y"] for s in te])
    t0 = time.time()
    for ep in range(args.epocas):
        certos = total = 0
        for x, y in J.lotes(c_tr, 48, seed=ep):
            y = y.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                saida = m(preparar(x))
                perda = nn.functional.cross_entropy(saida.float(), y, label_smoothing=0.1)
            otim.zero_grad(set_to_none=True)
            perda.backward()
            otim.step()
            if agenda.last_epoch < args.epocas * passos - 1:
                agenda.step()
            certos += (saida.argmax(1) == y).sum().item()
            total += y.numel()
        print(f"Esqueleto época {ep + 1}/{args.epocas}: treino {100 * certos / total:.1f}%, teste {100 * (prever() == Yt).mean():.1f}% "
              f"({time.time() - t0:.0f} s)", flush=True)
    p = prever()
    ok = p == Yt
    mao = np.array([s["present"][:, 0].mean() >= 0.7 for s in te])
    fonte = np.array([s["fonte"] for s in te])
    r = {"todos": round(100 * ok.mean(), 1), "com_mao": round(100 * ok[mao].mean(), 1),
         "por_fonte": {f: {"n": int((fonte == f).sum()), "todos": round(100 * ok[fonte == f].mean(), 1),
                           "com_mao": round(100 * ok[(fonte == f) & mao].mean(), 1) if ((fonte == f) & mao).any() else None}
                       for f in sorted(set(fonte))},
         "por_classe": {n: int(ok[Yt == idx[n]].sum()) for n in nomes if (Yt == idx[n]).any()}}
    (RAIZ / args.saida).write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESUMO", json.dumps({k: v for k, v in r.items() if k != "por_classe"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
