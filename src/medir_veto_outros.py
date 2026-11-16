"""Testa o veto da CNN sobre a contagem de dedos, com os gestos não usados juntos em "Outros gestos".

Ideia (16/11): a contagem de dedos confunde números com gestos do Jester feitos com dedos à mostra (zoom, deslizar com
dois dedos, girar a mão...). A CNN conhece esses gestos. Sem retreinar, as notas dos gestos que não são comando no
carro (e "Doing other things") são somadas numa classe "Outros gestos"; se a CNN apontar "Outros gestos" (ou um
gesto de movimento), o número da contagem é ignorado.

Mede no teste misto (mesmos dados e notas salvas de src/medir_modelo_ao_vivo.py), sem rodar rede:
  - sem veto (o modelo de hoje);
  - veto A: a classe de maior nota da CNN (36 classes, não usados somados em "Outros gestos") é "Outros gestos";
  - veto B (o usado ao vivo desde 16/11 à noite): é "Outros gestos" ou um gesto de movimento do carro;
  - veto C (testado): também "No gesture". Saiu porque a CNN lê números parados como "sem gesto" e vetava números certos.
Acerto em dois critérios: nas 36 classes (como as outras linhas da tabela) e com os gestos não usados juntos.
Para cada um: acerto geral, nos números e nos outros gestos; confiança da contagem por número (quando ela passa, em
quantos % era o número certo) e % das amostras sem número em que passou um número.

Uso:
    demo\\python.bat src\\medir_veto_outros.py
"""
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import comparar_misto as CM
import medir_confianca_numeros as MC

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "data/pontos/misto_modelos"
COMANDOS_CARRO = {"Swiping Left", "Swiping Right", "Thumb Up", "Thumb Down", "Drumming Fingers", "Stop Sign",
                  "Pushing Hand Away", "Pulling Hand In"}
FICAM = COMANDOS_CARRO | set(MC.NUMEROS) | {"No gesture"}   # o resto vai para "Outros gestos"


def main():
    jd = pickle.load(open(RAIZ / "data/pontos/jester_amostra.pkl", "rb"))
    nomes = list(jd["nomes"]) + CM.EXTRAS
    idx = {n: i for i, n in enumerate(nomes)}
    amostras = [{"fonte": "Jester", "classe": c["classe"], "divisao": "treino" if c["divisao"] == "treino" else "teste",
                 **{k: c[k] for k in ("img", "world", "present", "altura_largura")}} for c in jd["clipes"]]
    amostras += pickle.load(open(RAIZ / "data/pontos/misto.pkl", "rb"))
    te = [s for s in amostras if s["divisao"] == "teste"]
    y = np.load(PASTA / "teste_y.npy")
    probs = np.load(PASTA / "cnn_pre_probs.npy")
    assert len(te) == len(y) == len(probs) and all(idx[s["classe"]] == t for s, t in zip(te, y))

    outros = [i for i, n in enumerate(nomes) if n not in FICAM]
    movimento = [idx[n] for n in COMANDOS_CARRO]
    numeros = [idx[n] for n in MC.NUMEROS]
    # CNN com "Outros gestos": as notas dos não usados somadas numa coluna nova
    p_outros = probs[:, outros].sum(1)
    p_ficam = probs.copy()
    p_ficam[:, outros] = 0
    top = np.where(p_outros > p_ficam.max(1), -1, p_ficam.argmax(1))   # -1 = "Outros gestos"
    veto = {"sem veto": np.zeros(len(y), bool), "veto A: CNN diz Outros gestos": top == -1,
            "veto B (ao vivo): CNN diz Outros gestos ou um movimento": (top == -1) | np.isin(top, movimento),
            # o usado ao vivo (16/11): também veta quando a CNN diz "Sem gesto"
            "veto C (testado): Outros gestos, um movimento ou Sem gesto": (top == -1) | np.isin(top, movimento) | (top == idx["No gesture"])}

    regra = [MC.disparo(s) for s in te]
    sem_num = probs.copy()
    sem_num[:, numeros] = 0
    pred_cnn = sem_num.argmax(1)
    # a resposta certa, com os não usados juntos: acertar "Outros gestos" vale para qualquer gesto não usado
    y_junto = np.where(np.isin(y, outros), -1, y)
    e_num = np.isin(y, numeros)
    r = {"o_que_e": __doc__.split("\n\n")[0], "outros_gestos": [nomes[i] for i in outros], "variantes": {}}
    for nome, v in veto.items():
        passou = [None if (n is None or v[i]) else n for i, n in enumerate(regra)]
        pred = np.array([idx[n] if n else c for n, c in zip(passou, pred_cnn)])
        pred_junto = np.where(np.isin(pred, outros), -1, pred)
        certo = pred_junto == y_junto
        conf = {}
        for k in MC.NUMEROS:
            ditos = [i for i, n in enumerate(passou) if n == k]
            mostrados = [i for i in range(len(y)) if nomes[y[i]] == k]
            conf[MC.CURTO[k]] = {"confianca_pct": round(100 * np.mean([nomes[y[i]] == k for i in ditos]), 1) if ditos else None,
                                 "passou_quando_mostrado_pct": round(100 * np.mean([passou[i] == k for i in mostrados]), 1) if mostrados else None,
                                 "vezes": len(ditos)}
        sem = [i for i in range(len(y)) if not e_num[i]]
        r["variantes"][nome] = {
            "acerto_36_classes_pct": round(100 * (pred == y).mean(), 1),
            "acerto_pct": round(100 * certo.mean(), 1), "acerto_numeros_pct": round(100 * certo[e_num].mean(), 1),
            "acerto_outros_pct": round(100 * certo[~e_num].mean(), 1),
            "sem_numero_passou_numero_pct": round(100 * np.mean([passou[i] is not None for i in sem]), 1),
            "por_numero": conf,
            "falsos_vieram_de": dict(collections.Counter(nomes[y[i]] for i in sem if passou[i] is not None).most_common(6)),
        }
    saida = RAIZ / "resultados/pontos/veto_outros.json"
    saida.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    for nome, x in r["variantes"].items():
        print(f"== {nome}: acerto {x['acerto_pct']}% (números {x['acerto_numeros_pct']}%, outros {x['acerto_outros_pct']}%), "
              f"sem número que passou número {x['sem_numero_passou_numero_pct']}%")
        for k, c in x["por_numero"].items():
            print(f"     {k}: confiança {c['confianca_pct']}% ({c['vezes']} vezes), passou quando mostrado {c['passou_quando_mostrado_pct']}%")
        print("     falsos de:", x["falsos_vieram_de"])


if __name__ == "__main__":
    main()
