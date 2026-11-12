"""Peças comuns para refinar a classificação dos números a partir dos pontos já extraídos (src/extrair_pontos.py).

Não roda o MediaPipe: lê o cache em data/pontos/ e simula o mesmo DecisorNumeros das demos,
com o mesmo critério de src/medir_pontos_videos.py (falsos por minuto nos trechos sem gesto do IPN,
com 1,5 s de tolerância depois de um gesto; trechos de apontar com o número certo disparado).
"""
import collections
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pontos_mao as P

RAIZ = Path(__file__).resolve().parent.parent
CACHE = RAIZ / "data/pontos"
CLASSES = ("nenhum", "One Finger", "Two Fingers", "Three Fingers", "Four Fingers")
ALVO_IPN = {"B0A": "One Finger", "B0B": "Two Fingers"}
# Rótulo por quadro no IPN. Cliques com um/dois dedos (G01, G02, G08, G09) ficam de fora do treino: a forma da
# mão é a de um número, mas o gesto não é.
ROTULO_IPN = {"B0A": 1, "B0B": 2, "D0X": 0, "G03": 0, "G04": 0, "G05": 0, "G06": 0, "G07": 0, "G10": 0, "G11": 0}
# HaGRID: só as formas canônicas dos números contam como número; o resto é "nenhum". "point" fica de fora (ambíguo).
ROTULO_HAGRID = {"one": 1, "peace": 2, "two_up": 2, "three": 3, "three2": 3, "four": 4,
                 "fist": 0, "palm": 0, "stop": 0, "like": 0, "dislike": 0, "ok": 0, "call": 0, "rock": 0,
                 "mute": 0, "little_finger": 0, "no_gesture": 0}


def carregar():
    ipn = pickle.load(open(CACHE / "ipn.pkl", "rb"))
    curtos = pickle.load(open(CACHE / "curtos.pkl", "rb"))
    hagrid = pickle.load(open(CACHE / "hagrid.pkl", "rb"))
    return ipn, curtos, hagrid


# ---------- arrays por vídeo ----------

def arrays(quadros):
    """Empilha os quadros: tempo (N), mão achada (N), mundo (N,21,3), imagem (N,21,3), confiança (N), esquerda (N)."""
    n = len(quadros)
    t = np.array([q["t"] for q in quadros], dtype=np.float64)
    tem = np.array([q["mao"] for q in quadros], dtype=bool)
    mundo = np.zeros((n, 21, 3), np.float32)
    imagem = np.zeros((n, 21, 3), np.float32)
    conf = np.zeros(n, np.float32)
    esquerda = np.zeros(n, bool)
    for i, q in enumerate(quadros):
        if q["mao"]:
            mundo[i], imagem[i], conf[i], esquerda[i] = q["mundo"], q["imagem"], q["conf"], q["lado"] == "Left"
    return {"t": t, "tem": tem, "mundo": mundo, "imagem": imagem, "conf": conf, "esquerda": esquerda}


def medidas_vet(mundo):
    """Mesmas medidas de pontos_mao.medidas, para N quadros de uma vez."""
    p = mundo
    d = lambda a, b: np.linalg.norm(p[:, a] - p[:, b], axis=-1)

    def ang(u, v):
        cos = (u * v).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-9)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    m = {}
    for nome, (a, b, c, e) in zip(P.NOMES_DEDOS[1:], P.DEDOS[1:]):
        m[f"{nome}_reta"] = d(a, e) / np.maximum(d(a, b) + d(b, c) + d(c, e), 1e-9)
        m[f"{nome}_angulo"] = ang(p[:, a] - p[:, 0], p[:, e] - p[:, a])
    m["polegar_reta"] = d(2, 4) / np.maximum(d(2, 3) + d(3, 4), 1e-9)
    m["polegar_fora"] = d(4, 17) / np.maximum(d(5, 17), 1e-9)
    return m


def direcao_imagem(imagem, altura_largura):
    """Ângulo (graus) entre cada dedo (base→ponta, na imagem) e "para cima". 0 = dedo apontando para cima."""
    a, l = altura_largura if altura_largura else (1, 1)
    xy = imagem[..., :2] * np.array([l, a], np.float32)
    saida = []
    for (b, _, _, e) in P.DEDOS[1:]:
        v = xy[:, e] - xy[:, b]
        saida.append(np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1])))  # 0 = para cima, 180 = para baixo
    return np.stack(saida, 1)


def tamanho_mao(imagem, altura_largura):
    """Pulso → base do médio, em fração da altura do quadro."""
    a, l = altura_largura if altura_largura else (1, 1)
    xy = imagem[..., :2] * np.array([l, a], np.float32)
    return np.linalg.norm(xy[:, 9] - xy[:, 0], axis=-1) / a


# ---------- regras ----------

REGRAS_ISABELA = {"reta": 0.85, "angulo": 65.0, "polegar_reta": 0.95, "polegar_fora": 1.20,
                  "dobrado": None, "para_cima": None, "conf": 0.0, "tam_min": 0.0}


def poses_regras(A, m, direcao, tamanho, r):
    """Pose de cada quadro (índice em CLASSES, ou -1 = não é número / sem mão) pelas regras com os parâmetros r.

    Além das regras originais:
      - dobrado: os dedos não esticados precisam estar claramente dobrados (reta < dobrado); dedo "meio
        dobrado" deixa a pose ambígua, e pose ambígua não é número. É o caso da mão relaxada.
      - para_cima: os dedos esticados precisam apontar para cima na imagem (ângulo <= para_cima).
      - conf: confiança mínima do MediaPipe na mão.
      - tam_min: tamanho mínimo da mão no quadro (mãos pequenas, longe, são lidas pior).
    """
    n = len(A["t"])
    estic = np.zeros((n, 5), bool)
    estic[:, 0] = (m["polegar_reta"] >= r["polegar_reta"]) & (m["polegar_fora"] >= r["polegar_fora"])
    for k, nome in enumerate(P.NOMES_DEDOS[1:], 1):
        estic[:, k] = (m[f"{nome}_reta"] >= r["reta"]) & (m[f"{nome}_angulo"] <= r["angulo"])
    total = estic.sum(1)
    pose = np.where((total >= 1) & (total <= 4) & ~((total == 1) & estic[:, 0]), total, -1)
    ok = A["tem"] & (A["conf"] >= r["conf"]) & (tamanho >= r["tam_min"])
    if r["dobrado"] is not None:
        for k, nome in enumerate(P.NOMES_DEDOS[1:], 1):
            ambiguo = ~estic[:, k] & (m[f"{nome}_reta"] >= r["dobrado"]) & (m[f"{nome}_angulo"] <= r["angulo"])
            ok &= ~ambiguo
    if r["para_cima"] is not None:
        for k in range(1, 5):
            ok &= ~estic[:, k] | (direcao[:, k - 1] <= r["para_cima"])
    return np.where(ok, pose, -1)


# ---------- decisor e medidas ----------

class _Mao:
    __slots__ = ("imagem", "mundo", "pose", "dedos")


def simular(A, poses, opcoes, altura_largura):
    """Roda o DecisorNumeros com a pose já decidida de cada quadro. Devolve [(t, nome do número)]."""
    a, l = altura_largura if altura_largura else (1, 1)
    d = P.DecisorNumeros(classificador=lambda m: m.pose, **opcoes)
    disparos = []
    for i in range(len(A["t"])):
        mao = None
        if A["tem"][i]:
            mao = _Mao()
            mao.imagem, mao.mundo, mao.dedos = A["imagem"][i], A["mundo"][i], None
            mao.pose = CLASSES[poses[i]] if poses[i] > 0 else None
        n = d.atualizar(mao, float(A["t"][i]), l, a)
        if n:
            disparos.append((float(A["t"][i]), n))
    return disparos


def rotulo_ipn(trechos, fps, t):
    f = t * fps + 1
    for ini, fim, r in trechos:
        if ini <= f <= fim:
            if r == "D0X" and any(x != "D0X" and fim2 < f <= fim2 + 1.5 * fps for _, fim2, x in trechos):
                return "logo após um gesto"
            return r
    return None


def medir_ipn(videos, disparos_por_video):
    """videos: {nome: registro do cache}; disparos_por_video: {nome: [(t, número)]}."""
    sem_gesto = falsos = 0.0
    falsos_classe = collections.Counter()
    apontar = {r: {"trechos": 0, "certos": 0} for r in ALVO_IPN}
    for v, R in videos.items():
        fps, trechos, disp = R["fps"], R["trechos"], disparos_por_video[v]
        sem_gesto += sum(f - i + 1 for i, f, r in trechos if r == "D0X") / fps / 60
        for t, n in disp:
            if rotulo_ipn(trechos, fps, t) == "D0X":
                falsos += 1
                falsos_classe[n] += 1
        for ini, fim, r in trechos:
            if r in ALVO_IPN:
                a, b = (ini - 1) / fps, (fim - 1) / fps
                apontar[r]["trechos"] += 1
                apontar[r]["certos"] += any(a <= t <= b + 0.5 and n == ALVO_IPN[r] for t, n in disp)
    pct = {ALVO_IPN[r]: round(100 * x["certos"] / max(x["trechos"], 1), 1) for r, x in apontar.items()}
    return {"falsos_por_minuto": round(falsos / max(sem_gesto, 1e-9), 3), "falsos": int(falsos),
            "minutos_sem_gesto": round(sem_gesto, 1), "falsos_por_classe": dict(falsos_classe.most_common()),
            "pct_trechos_certos": pct, "media_certos": round(float(np.mean(list(pct.values()))), 1)}


def medir_curtos(itens, disparos_por_item):
    saida = {}
    sem = [(it, d) for it, d in zip(itens, disparos_por_item) if it["base"] == "Slovo sem sinal"]
    if sem:
        minutos = sum(it["duracao"] for it, _ in sem) / 60
        n = sum(len(d) for _, d in sem)
        saida["Slovo sem sinal"] = {"videos": len(sem), "minutos": round(minutos, 1), "falsos": n, "falsos_por_minuto": round(n / minutos, 3)}
    for base in ("Slovo sinais", "ASL Citizen sinais"):
        for classe in ("One Finger", "Two Fingers"):
            g = [(it, d) for it, d in zip(itens, disparos_por_item) if it["base"] == base and it["classe"] == classe]
            if not g:
                continue
            certos = 0
            for it, d in g:
                if it["inicio"] is not None and it["fim"]:
                    fps = it["fps"]
                    d = [x for x in d if it["inicio"] / fps - 0.5 <= x[0] <= it["fim"] / fps + 1.0]
                certos += any(n == classe for _, n in d)
            saida.setdefault(base, {})[classe] = round(100 * certos / len(g), 1)
    return saida
