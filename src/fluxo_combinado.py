"""Falsos disparos em vídeo contínuo: CNN pré-treinada, GRU sobre pontos e a combinação das duas (média 0,7/0,3).

Mesmo laço e mesmo critério de src/fluxo_ipn.py: quadros a 12 por segundo, janelas de 1, 1,5 e 3 s, uma análise
a cada 0,15 s, média das 2 últimas análises e o gatilho das demos (>= 0,7 em 3 análises seguidas, pausa de 1,2 s).
Em cada análise, cada modelo dá as notas das 36 classes em cada janela; a combinação faz a média por janela; fica a
janela com o gesto mais confiante, como no player.

Modelos (de src/comparar_misto.py, em data/pontos/misto_modelos/): cnn_pre_final.pt e gru_final.pt.
Pontos: o cache de src/extrair_pontos.py (MediaPipe a 15 quadros/s, reamostrado para 12).

Vídeos: IPN, as 13 pessoas de validação (52 vídeos), e Slovo "sem sinal" de teste (falsos por minuto).

Uso:
    demo\\python.bat src\\fluxo_combinado.py
"""
import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import fluxo_ipn as F
import modelos as M
import treinar_jester as T
import treinar_pontos_jester as TP

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "data/pontos/misto_modelos"
QPS, N, JANELAS, INTERVALO = 12, 16, (1.0, 1.5, 3.0), 0.15


def pontos_12(registro):
    """Cache (15 quadros/s, uma mão) -> arrays no formato da GRU, a 12 quadros/s, cobrindo o vídeo inteiro."""
    q = registro["quadros"]
    t = np.array([x["t"] for x in q])
    n = int(np.floor(t[-1] * QPS)) + 1 if len(t) else 0
    alvo = np.arange(n) / QPS
    idx = np.clip(np.searchsorted(t, alvo), 0, len(t) - 1)
    img = np.zeros((n, 2, 21, 3), np.float32)
    world = np.zeros((n, 2, 21, 3), np.float32)
    pres = np.zeros((n, 2), bool)
    score = np.zeros((n, 2), np.float32)
    for k, i in enumerate(idx):
        if q[i]["mao"]:
            img[k, 0], world[k, 0], pres[k, 0], score[k, 0] = q[i]["imagem"], q[i]["mundo"], True, q[i]["conf"]
    return {"img": img, "world": world, "present": pres, "score": score, "altura_largura": registro["altura_largura"]}


@torch.no_grad()
def analisar(cnn, gru, norm, quadros, pontos, gestos):
    """Para cada análise: notas por janela da CNN e da GRU. Devolve tempos e listas [análise][janela] de (p_cnn, p_gru)."""
    passo = INTERVALO * QPS
    t = min(int(round(j * QPS)) for j in JANELAS)
    tempos, lotes_c, lotes_g, por = [], [], [], []
    xg = norm(TP.atributos(pontos)[0]) if pontos is not None else None
    while t <= len(quadros):
        k = 0
        for d in JANELAS:
            n = int(round(d * QPS))
            if t >= n:
                lotes_c.append(quadros[np.round(np.linspace(t - n, t - 1, N)).astype(int)])
                if xg is not None:
                    fim = min(t, len(xg))
                    lotes_g.append(TP.fixar(xg[max(0, fim - n):fim])) if fim > 0 else lotes_g.append(TP.fixar(np.zeros((1, xg.shape[1]), np.float32)))
                k += 1
        por.append(k)
        tempos.append(t / QPS)
        t = int(round(t + passo))
    pc = []
    for i in range(0, len(lotes_c), 96):
        x = torch.from_numpy(np.stack(lotes_c[i:i + 96]).astype(np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pc.append(cnn(T.preparar(x, "cuda")).float().softmax(1).cpu().numpy())
    pc = np.concatenate(pc)
    if lotes_g:
        pg = []
        for i in range(0, len(lotes_g), 512):
            xs, ms = zip(*lotes_g[i:i + 512])
            pg.append(gru(torch.from_numpy(np.stack(xs)).cuda(), torch.from_numpy(np.stack(ms)).cuda()).float().softmax(1).cpu().numpy())
        pg = np.concatenate(pg)
    else:
        pg = np.full_like(pc, 1 / pc.shape[1])
    return np.array(tempos), por, pc, pg


def escolher(por, p, gestos):
    """Fica, em cada análise, a janela com o gesto mais confiante (como no player)."""
    saida, k = [], 0
    for n in por:
        b = p[k:k + n]
        saida.append(b[int(np.argmax(b[:, gestos].max(1)))])
        k += n
    return np.stack(saida)


def disparos(tempos, probs, nomes):
    g, hist, saida = F.Gatilho(), collections.deque(maxlen=2), []
    for t, p in zip(tempos, probs):
        hist.append(p)
        m = np.mean(hist, axis=0)
        k = int(m.argmax())
        d = g.atualizar(nomes[k], float(m[k]), t)
        if d:
            saida.append((t, d))
    return saida


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--peso-gru", type=float, default=0.3)
    a.add_argument("--peso-gru-numeros", type=float, nargs="*", default=[None, 0.6, 0.8, 1.0],
                   help="peso da GRU nas classes de número (None = igual ao peso geral); cada valor é uma regra testada")
    a.add_argument("--saida", default="resultados/pontos/fluxo_combinado.json")
    args = a.parse_args()
    e = torch.load(PASTA / "cnn_pre_final.pt", map_location="cuda", weights_only=False)
    nomes = list(e["nomes"])
    cnn = M.criar_modelo("cnn_3d", len(nomes)).cuda().eval()
    cnn.load_state_dict(e["modelo"])
    gq = torch.load(PASTA / "gru_final.pt", map_location="cuda", weights_only=False)
    gru = TP.Rede(gq["entrada"], len(nomes)).cuda().eval()
    gru.load_state_dict(gq["modelo"])
    media, desvio = gq["media"], gq["desvio"]
    norm = lambda x: np.where(x[:, -2:-1] > 0, (x - media) / desvio, 0).astype(np.float32)
    gestos = [i for i, n in enumerate(nomes) if n not in T.SEM_GESTO]
    w = args.peso_gru
    numeros = [nomes.index(n) for n in ("One Finger", "Two Fingers", "Three Fingers", "Four Fingers")]
    regras = {}
    for wn in args.peso_gru_numeros:
        pesos = np.full(len(nomes), w, np.float32)
        if wn is not None:
            pesos[numeros] = wn
        rotulo = f"Combinação ({w:.1f} GRU" + (f"; números {wn:.1f} GRU)" if wn is not None else ")")
        regras[rotulo] = pesos

    def combinar(pc, pg, pesos):
        p = (1 - pesos) * pc + pesos * pg
        return p / p.sum(1, keepdims=True)
    MODELOS = ("CNN pré-treinada", "GRU sobre pontos", *regras)

    ipn = pickle.load(open(RAIZ / "data/pontos/ipn.pkl", "rb"))
    videos = sorted(v for v, r in ipn.items() if r["divisao"] == "test")
    R = {m: {"falsos": collections.Counter(), "disparos": collections.Counter(), "apontar": {"B0A": 0, "B0B": 0}, "trechos_certos": {"B0A": 0, "B0B": 0}}
         for m in MODELOS}
    sem_gesto = 0.0
    trechos_apontar = {"B0A": 0, "B0B": 0}
    alvo = {"B0A": "One Finger", "B0B": "Two Fingers"}
    for k, v in enumerate(videos, 1):
        reg = ipn[v]
        quadros, fps = F.quadros_12(RAIZ / f"data/IPN/videos/videos/{v}.avi")
        tempos, por, pc, pg = analisar(cnn, gru, norm, quadros, pontos_12(reg), gestos)
        trechos = reg["trechos"]
        sem_gesto += sum(f - i + 1 for i, f, r in trechos if r == "D0X") / fps / 60
        for r in alvo:
            trechos_apontar[r] += sum(1 for *_, x in trechos if x == r)
        for nome, p in zip(MODELOS, (pc, pg, *[combinar(pc, pg, r) for r in regras.values()])):
            disp = disparos(tempos, escolher(por, p, gestos), nomes)
            for t, d in disp:
                rot = F_rotulo(trechos, fps, t)
                R[nome]["disparos"][rot or "entre trechos"] += 1
                if rot == "D0X":
                    R[nome]["falsos"][d] += 1
            for ini, fim, r in trechos:
                if r in alvo:
                    a0, b0 = (ini - 1) / fps, (fim - 1) / fps
                    R[nome]["trechos_certos"][r] += any(a0 <= t <= b0 + 0.5 and d == alvo[r] for t, d in disp)
        print(f"  IPN {k}/{len(videos)}", flush=True)

    curtos = pickle.load(open(RAIZ / "data/pontos/curtos.pkl", "rb"))
    slovo = [c for c in curtos if c["base"] == "Slovo sem sinal" and c["divisao"] == "test"]
    S = {m: collections.Counter() for m in MODELOS}
    minutos_slovo = 0.0
    for c in slovo:
        quadros, _ = F.quadros_12(RAIZ / c["caminho"])
        if len(quadros) < QPS:
            continue
        minutos_slovo += len(quadros) / QPS / 60
        tempos, por, pc, pg = analisar(cnn, gru, norm, quadros, pontos_12(c), gestos)
        for nome, p in zip(MODELOS, (pc, pg, *[combinar(pc, pg, r) for r in regras.values()])):
            S[nome].update(d for _, d in disparos(tempos, escolher(por, p, gestos), nomes))

    saida = {"peso_gru": w, "ipn_minutos_sem_gesto": round(sem_gesto, 1), "slovo_minutos": round(minutos_slovo, 1), "modelos": {}}
    for nome in MODELOS:
        f = sum(R[nome]["falsos"].values())
        saida["modelos"][nome] = {
            "ipn_falsos_por_minuto": round(f / sem_gesto, 3), "ipn_falsos": f, "ipn_falsos_por_classe": dict(R[nome]["falsos"].most_common()),
            "ipn_um_dedo_trechos_certos": f"{R[nome]['trechos_certos']['B0A']}/{trechos_apontar['B0A']}",
            "ipn_dois_dedos_trechos_certos": f"{R[nome]['trechos_certos']['B0B']}/{trechos_apontar['B0B']}",
            "slovo_falsos_por_minuto": round(sum(S[nome].values()) / max(minutos_slovo, 1e-9), 3),
            "slovo_falsos_por_classe": dict(S[nome].most_common()),
            "disparos_por_trecho": dict(R[nome]["disparos"].most_common())}
    (RAIZ / args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    for nome, s in saida["modelos"].items():
        print(f"\n== {nome}: IPN {s['ipn_falsos_por_minuto']} falsos/min ({s['ipn_falsos']} em {saida['ipn_minutos_sem_gesto']} min) {s['ipn_falsos_por_classe']}"
              f" | 1 dedo {s['ipn_um_dedo_trechos_certos']} | 2 dedos {s['ipn_dois_dedos_trechos_certos']}"
              f" | Slovo {s['slovo_falsos_por_minuto']} falsos/min {s['slovo_falsos_por_classe']}")


def F_rotulo(trechos, fps, seg):
    f = seg * fps + 1
    for ini, fim, r in trechos:
        if ini <= f <= fim:
            if r == "D0X" and any(x != "D0X" and fim2 < f <= fim2 + 1.5 * fps for _, fim2, x in trechos):
                return "logo após um gesto"
            return r
    return None


if __name__ == "__main__":
    main()
