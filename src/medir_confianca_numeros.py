"""Mede a confiança da contagem de dedos (a regra que decide os números ao vivo) em dados de teste.

Por quê: ao vivo, quando a mão está parada com 1 a 4 dedos, o número é dado pela contagem geométrica dos pontos do
MediaPipe (src/pontos_mao.py). A contagem não dá porcentagem; o player usava uma nota fixa de 0,9, escolhida sem medir.
Aqui a nota passa a vir de um teste: quando a regra diz "k dedos", em quantos % das vezes eram k dedos de verdade.

Teste, sem treino: nada é ajustado aqui. Os dados são só de pessoas fora do treino e fora do ajuste dos limiares:
  - conjunto misto, divisão "teste" (data/pontos/misto.pkl): IPN (pessoas de validação), Slovo (teste), DriverMHG
    (pessoa 11), HaGRID (validação, só a metade de pessoas que NÃO foi usada para ajustar os limiares das regras);
  - Jester, validação oficial (data/pontos/jester_amostra.pkl): 27 gestos sem número (deslizar, pare, polegar...),
    para ver quantas vezes a regra inventa um número.

Simulação igual ao player (demo/player.py, Pontos.mao_parada e Pontos.contagem_estavel): 12 quadros por segundo, uma
análise a cada 2 quadros (~0,17 s; ao vivo, 0,15 s), mão parada = presente em 5 dos últimos 6 quadros e pulso andou
no máximo 0,5 palma; número = mesma contagem de 1 a 4 dedos em 5 dos 6 quadros; dispara na primeira análise em que isso acontece
(como o player desde 16/11, sem nota e sem as 3 análises seguidas da CNN). Cada amostra conta uma vez: o primeiro número disparado (ou nenhum).

Uso:
    demo\\python.bat src\\medir_confianca_numeros.py
"""
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pontos_mao as P

RAIZ = Path(__file__).resolve().parent.parent
NUMEROS = list(P.NUMEROS)
CURTO = {"One Finger": "1 dedo", "Two Fingers": "2 dedos", "Three Fingers": "3 dedos", "Four Fingers": "4 dedos"}
N, MINIMO, PASSO, SEGUIDAS, MOVIMENTO = 6, 5, 2, 1, 0.5  # ao vivo (desde 16/11), o número dispara direto: SEGUIDAS = 1


def metade(user_id):  # a mesma divisão de src/avaliar_pontos_hagrid.py: metade 0 ajustou os limiares
    return int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 2


def parada(img, pres, al, t):
    q = range(t - N + 1, t + 1)
    if sum(bool(pres[i]) for i in q) < N - 1:
        return False
    a, l = al
    xy = [img[i, :, :2] * np.array([l, a], np.float32) for i in q if pres[i]]
    palma = max(float(np.linalg.norm(xy[-1][9] - xy[-1][0])), 1e-6)
    return max(float(np.linalg.norm(p[0] - xy[-1][0])) for p in xy) / palma <= MOVIMENTO


def contagem(world, pres, t):
    poses = [(P.classificar_pose(P.contar_dedos(world[i])) or "nao_numero") if pres[i] else None for i in range(t - N + 1, t + 1)]
    validas = [q for q in poses if q is not None]
    if not validas:
        return None
    mais, vezes = collections.Counter(validas).most_common(1)[0]
    return mais if vezes >= MINIMO else None


def disparo(amostra):
    """Primeiro número disparado na amostra (ou None)."""
    img, world, pres, al = amostra["img"][:, 0], amostra["world"][:, 0], amostra["present"][:, 0], amostra["altura_largura"]
    atual, seguidas = None, 0
    for t in range(N - 1, len(pres), PASSO):
        num = contagem(world, pres, t) if parada(img, pres, al, t) else None
        num = num if num in NUMEROS else None
        if num is None:
            atual, seguidas = None, 0
            continue
        seguidas = seguidas + 1 if num == atual else 1
        atual = num
        if seguidas >= SEGUIDAS:
            return num
    return None


def main():
    amostras = []
    for a in pickle.load(open(RAIZ / "data/pontos/misto.pkl", "rb")):
        if a["divisao"] != "teste":
            continue
        if a["fonte"] == "HaGRID" and metade(a["pessoa"]) == 0:
            continue
        amostras.append({"fonte": a["fonte"], "classe": a["classe"], **{k: a[k] for k in ("img", "world", "present", "altura_largura")}})
    j = pickle.load(open(RAIZ / "data/pontos/jester_amostra.pkl", "rb"))
    for a in j["clipes"]:
        if a["divisao"] == "validacao":
            amostras.append({"fonte": "Jester", "classe": a["classe"], **{k: a[k] for k in ("img", "world", "present", "altura_largura")}})

    linhas = []
    for a in amostras:
        linhas.append((a["fonte"], a["classe"], disparo(a)))

    disparos = collections.Counter(d for _, _, d in linhas if d)
    por_numero = {}
    for k in NUMEROS:
        verdadeiros = [d for f, c, d in linhas if c == k]
        disparados = [(f, c) for f, c, d in linhas if d == k]
        certos = sum(c == k for _, c in disparados)
        por_numero[CURTO[k]] = {
            "amostras_desse_numero": len(verdadeiros),
            "disparou_certo_pct": round(100 * sum(d == k for d in verdadeiros) / max(len(verdadeiros), 1), 1),
            "disparou_outro_numero_pct": round(100 * sum(d not in (None, k) for d in verdadeiros) / max(len(verdadeiros), 1), 1),
            "nao_disparou_pct": round(100 * sum(d is None for d in verdadeiros) / max(len(verdadeiros), 1), 1),
            "vezes_que_a_regra_disse_esse_numero": len(disparados),
            "confianca_pct": round(100 * certos / len(disparados), 1) if disparados else None,
            "erros_vinham_de": dict(collections.Counter(f"{f}: {CURTO.get(c, c)}" for f, c in disparados if c != k).most_common(8)),
        }
    sem_numero = [(f, c, d) for f, c, d in linhas if c not in NUMEROS]
    falsos_por_fonte = {}
    for fonte in sorted({f for f, _, _ in sem_numero}):
        g = [d for f, _, d in sem_numero if f == fonte]
        falsos_por_fonte[fonte] = {"amostras": len(g), "disparou_numero_pct": round(100 * sum(d is not None for d in g) / len(g), 1)}
    falsos_por_classe = collections.Counter(f"{f}: {c}" for f, c, d in sem_numero if d)
    comandos = [por_numero[CURTO[k]]["confianca_pct"] for k in ("One Finger", "Two Fingers", "Four Fingers")]
    r = {
        "o_que_e": "Confiança da contagem de dedos ao vivo: quando a regra dispara 'k dedos', % das vezes em que eram k dedos. "
                   "Teste sem treino, pessoas fora do treino e do ajuste dos limiares. Simulação do player: 12 qps, análise a "
                   "cada 2 quadros, mão parada (5 de 6 quadros, pulso <= 0,5 palma), mesma contagem em 5 de 6; dispara direto, sem as 3 análises seguidas.",
        "amostras": dict(collections.Counter(f"{f}" for f, _, _ in linhas)),
        "por_numero": por_numero,
        "confianca_media_1_2_4_dedos_pct": round(float(np.mean([c for c in comandos if c is not None])), 1),
        "abaixo_de_70": [n for n, v in por_numero.items() if v["confianca_pct"] is not None and v["confianca_pct"] < 70],
        "amostras_sem_numero": {"total": len(sem_numero), "disparou_numero_pct": round(100 * sum(d is not None for _, _, d in sem_numero) / len(sem_numero), 1),
                                "por_fonte": falsos_por_fonte, "classes_que_mais_dispararam": dict(falsos_por_classe.most_common(10))},
    }
    saida = RAIZ / "resultados/pontos/confianca_numeros.json"
    saida.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
