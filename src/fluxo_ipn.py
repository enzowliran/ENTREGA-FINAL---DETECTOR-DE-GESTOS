"""Falsos disparos em vídeo real e contínuo: os vídeos de validação do IPN Hand rodados como se fossem a webcam.

São 52 vídeos de 13 pessoas que não entram no treino, cada um com ~2 min de gestos e intervalos sem gesto,
gravados em webcam de notebook. O laço é o mesmo de demo/player.py: quadros a 12 por segundo, janelas de 1, 1,5 e 3 s,
uma análise a cada 0,15 s, média das 2 últimas e o gatilho (confiança >= 0,7 em 3 análises seguidas, pausa de 1,2 s).

Cada disparo é classificado pelo trecho anotado em que acontece:
  - em "sem gesto" (D0X): falso disparo;
  - em apontar com um/dois dedos: certo se for o dedo certo;
  - nos outros gestos do IPN: anotado à parte (muitos se parecem com gestos do Jester).

Uso:
    python src/fluxo_ipn.py --modelos resultados/jester/cnn_3d_mais_gestos/melhor.pt resultados/jester/cnn_3d_dedos/final.pt
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import modelos as M
import treinar_jester as T

RAIZ = Path(__file__).resolve().parent.parent
QPS, N, LADO, JANELAS, INTERVALO = 12, 16, 96, (1.0, 1.5, 3.0), 0.15


class Gatilho:
    """Igual ao de demo/player.py, sem repetição."""

    def __init__(self, limiar=0.7, seguidas=3, pausa=1.2):
        self.limiar, self.seguidas, self.pausa = limiar, seguidas, pausa
        self.atual, self.contagem, self.fim_pausa, self.armado = None, 0, 0.0, True

    def atualizar(self, nome, conf, agora):
        if nome in T.SEM_GESTO or conf < self.limiar:
            self.atual, self.contagem, self.armado = None, 0, True
            return None
        if nome != self.atual:
            self.atual, self.contagem, self.armado = nome, 0, True
        self.contagem += 1
        if self.armado and self.contagem >= self.seguidas and agora >= self.fim_pausa:
            self.armado, self.fim_pausa = False, agora + self.pausa
            return nome
        return None


def quadros_12(caminho):
    """Todos os quadros a 12 por segundo, já no recorte da webcam (quadrado central, 96x96, RGB)."""
    cap, saida, i, proximo = cv2.VideoCapture(str(caminho)), [], 0, 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while True:
        ok, q = cap.read()
        if not ok:
            break
        if i >= proximo:
            a, l = q.shape[:2]
            b = min(a, l)
            r = cv2.resize(q[(a - b) // 2:(a - b) // 2 + b, (l - b) // 2:(l - b) // 2 + b], (LADO, LADO), interpolation=cv2.INTER_LINEAR)
            saida.append(r[..., ::-1].transpose(2, 0, 1))
            proximo += fps / QPS
        i += 1
    cap.release()
    return np.stack(saida), fps


@torch.no_grad()
def probabilidades(modelo, quadros, indices_gesto):
    """Uma análise a cada 0,15 s de vídeo, como no player: devolve (tempos, probabilidades da janela escolhida)."""
    tempos, lotes, por_analise = [], [], []
    passo_q = INTERVALO * QPS
    t = min(int(round(j * QPS)) for j in JANELAS)
    while t <= len(quadros):
        janelas = []
        for d in JANELAS:
            n = int(round(d * QPS))
            if t >= n:
                idx = np.round(np.linspace(t - n, t - 1, N)).astype(int)
                janelas.append(quadros[idx])
        lotes += janelas
        por_analise.append(len(janelas))
        tempos.append(t / QPS)
        t = int(round(t + passo_q)) if passo_q >= 1 else t + 1
    saidas = []
    for k in range(0, len(lotes), 96):
        x = torch.from_numpy(np.stack(lotes[k:k + 96]).astype(np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            saidas.append(modelo(T.preparar(x, "cuda")).float().softmax(1).cpu().numpy())
    todas = np.concatenate(saidas)
    escolhidas, k = [], 0
    for n in por_analise:  # como no player: fica a janela com o gesto mais confiante
        bloco = todas[k:k + n]
        escolhidas.append(bloco[int(np.argmax(bloco[:, indices_gesto].max(1)))])
        k += n
    return np.array(tempos), np.stack(escolhidas)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--modelos", nargs="+", required=True)
    a.add_argument("--saida", default="resultados/dedos/fluxo_ipn.json")
    args = a.parse_args()
    meta = {l["Video Name"]: l["Set"] for l in csv.DictReader(open(RAIZ / "data/IPN/anotacoes/metadata.csv", encoding="utf-8"))}
    anot = collections.defaultdict(list)
    for l in csv.DictReader(open(RAIZ / "data/IPN/anotacoes/Annot_List.txt", encoding="utf-8")):
        anot[l["video"]].append((int(l["t_start"]), int(l["t_end"]), l["label"]))
    videos = sorted(v for v in anot if meta.get(v) == "test")
    print(f"{len(videos)} vídeos de validação (pessoas fora do treino)", flush=True)

    modelos = []
    for caminho in args.modelos:
        e = torch.load(RAIZ / caminho, map_location="cuda", weights_only=False)
        m = M.criar_modelo(e["arquitetura"], len(e["nomes"])).cuda().eval()
        m.load_state_dict(e["modelo"])
        nomes = list(e["nomes"])
        modelos.append((caminho, m, nomes, [i for i, n in enumerate(nomes) if n not in T.SEM_GESTO]))

    resultado = {c: {"minutos": 0.0, "minutos_sem_gesto": 0.0, "disparos": collections.Counter(), "falsos_por_classe": collections.Counter(),
                     "apontar": {"B0A": collections.Counter(), "B0B": collections.Counter()}, "trechos_apontar": {"B0A": 0, "B0B": 0}} for c, *_ in modelos}
    for k, video in enumerate(videos, 1):
        quadros, fps = quadros_12(RAIZ / f"data/IPN/videos/videos/{video}.avi")
        trechos = anot[video]

        def rotulo(seg):  # trecho anotado no instante seg (anotação em quadros de 30 fps, começando em 1)
            f = seg * fps + 1
            for ini, fim, r in trechos:
                if ini <= f <= fim:
                    # o gatilho dispara com um pequeno atraso: até 1,5 s depois de um gesto não conta como falso
                    if r == "D0X" and any(x != "D0X" and fim2 < f <= fim2 + 1.5 * fps for _, fim2, x in trechos):
                        return "logo após um gesto"
                    return r
            return None

        for caminho, m, nomes, ig in modelos:
            R = resultado[caminho]
            tempos, probs = probabilidades(m, quadros, ig)
            R["minutos"] += len(quadros) / QPS / 60
            R["minutos_sem_gesto"] += sum(f - i + 1 for i, f, r in trechos if r == "D0X") / fps / 60
            for r in ("B0A", "B0B"):
                R["trechos_apontar"][r] += sum(1 for _, _, x in trechos if x == r)
            g, hist = Gatilho(), collections.deque(maxlen=2)
            for t, p in zip(tempos, probs):
                hist.append(p)
                media = np.mean(hist, axis=0)
                melhor = int(media.argmax())
                disparo = g.atualizar(nomes[melhor], float(media[melhor]), t)
                if not disparo:
                    continue
                r = rotulo(t)
                R["disparos"][r or "entre trechos"] += 1
                if r == "D0X":
                    R["falsos_por_classe"][disparo] += 1
                elif r in ("B0A", "B0B"):
                    R["apontar"][r][disparo] += 1
        print(f"  {k}/{len(videos)} vídeos", flush=True)

    saida = {}
    for caminho, R in resultado.items():
        falsos = sum(R["falsos_por_classe"].values())
        acertos = {r: R["apontar"][r].get(alvo, 0) for r, alvo in (("B0A", "One Finger"), ("B0B", "Two Fingers"))}
        saida[caminho] = {
            "minutos_de_video": round(R["minutos"], 1), "minutos_sem_gesto": round(R["minutos_sem_gesto"], 1),
            "falsos_em_sem_gesto": falsos, "falsos_por_minuto_sem_gesto": round(falsos / R["minutos_sem_gesto"], 3),
            "falsos_por_classe": dict(R["falsos_por_classe"].most_common()),
            "disparos_por_trecho": dict(R["disparos"].most_common()),
            "apontar_um_dedo": {"trechos": R["trechos_apontar"]["B0A"], "disparos_certos": acertos["B0A"], "disparos": dict(R["apontar"]["B0A"].most_common(5))},
            "apontar_dois_dedos": {"trechos": R["trechos_apontar"]["B0B"], "disparos_certos": acertos["B0B"], "disparos": dict(R["apontar"]["B0B"].most_common(5))},
        }
        s = saida[caminho]
        print(f"\n== {caminho}\n  {s['minutos_de_video']} min de vídeo, {s['minutos_sem_gesto']} min sem gesto: {falsos} falsos "
              f"({s['falsos_por_minuto_sem_gesto']} por minuto) {s['falsos_por_classe']}\n"
              f"  apontar 1 dedo: {acertos['B0A']} disparos certos em {s['apontar_um_dedo']['trechos']} trechos {s['apontar_um_dedo']['disparos']}\n"
              f"  apontar 2 dedos: {acertos['B0B']} disparos certos em {s['apontar_dois_dedos']['trechos']} trechos {s['apontar_dois_dedos']['disparos']}")
    destino = RAIZ / args.saida
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvo em {destino}")


if __name__ == "__main__":
    main()
