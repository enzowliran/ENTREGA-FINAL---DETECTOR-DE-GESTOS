"""Acerto do modelo final (CNN 3D + contagem de dedos, sem veto) só nos gestos que viram comando no carro.

Por quê: o acerto de 70,9% do teste misto conta as 36 classes, e 25 delas não são comando no carro. Para a
apresentação, o que interessa é: quando a pessoa faz um dos 11 gestos do carro, em quantos % o modelo acerta.

Mesmos dados e notas salvas de src/medir_veto_outros.py (teste misto, pessoas fora do treino), sem rodar rede:
  - a contagem de dedos é simulada como no player (mão parada, mesmo número em 5 de 6 quadros);
  - se ela dá um número, vale o número; senão, vale a classe de maior nota da CNN, sem as classes de número.

Uso:
    demo\\python.bat src\\medir_gestos_do_carro.py
"""
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
MOVIMENTOS = ["Swiping Right", "Swiping Left", "Thumb Up", "Thumb Down", "Stop Sign", "Pushing Hand Away",
              "Drumming Fingers", "Pulling Hand In"]
NUMEROS_CARRO = ["One Finger", "Two Fingers", "Four Fingers"]
# o mesmo mapa do carro 3D (demo/player.py, COMANDOS_CARRO3D): gesto -> comando
COMANDO = {"Swiping Left": 0, "Swiping Right": 1, "Thumb Up": 2, "Thumb Down": 3, "Drumming Fingers": 4, "Stop Sign": 5,
           "Pushing Hand Away": 5, "Pulling Hand In": 6, "One Finger": 7, "Two Fingers": 8, "Four Fingers": 9}


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

    numeros = [idx[n] for n in MC.NUMEROS]
    regra = [MC.disparo(s) for s in te]
    sem_num = probs.copy()
    sem_num[:, numeros] = 0
    pred_cnn = sem_num.argmax(1)
    pred = np.array([idx[n] if n else c for n, c in zip(regra, pred_cnn)])   # modelo final: o número tem prioridade
    so_cnn = probs.argmax(1)                                                   # a CNN sozinha, com as 36 classes

    # comando que cada classe dispara no carro (pare e empurrar a mão dão o mesmo comando); -1 = nenhum
    cmd = np.full(len(nomes), -1)
    for n, c in COMANDO.items():
        cmd[idx[n]] = c

    def bloco(classes):
        sel = np.isin(y, [idx[c] for c in classes])
        return {"amostras": int(sel.sum()), "acerto_pct": round(100 * float((pred[sel] == y[sel]).mean()), 1),
                "comando_certo_pct": round(100 * float((cmd[pred[sel]] == cmd[y[sel]]).mean()), 1),
                "cnn_sozinha_pct": round(100 * float((so_cnn[sel] == y[sel]).mean()), 1)}

    por_gesto = {}
    for c in NUMEROS_CARRO + MOVIMENTOS:
        sel = y == idx[c]
        erros = {}
        for p in pred[sel & (pred != y)]:
            erros[nomes[p]] = erros.get(nomes[p], 0) + 1
        por_gesto[c] = {"amostras": int(sel.sum()), "acerto_pct": round(100 * float((pred[sel] == y[sel]).mean()), 1),
                        "comando_certo_pct": round(100 * float((cmd[pred[sel]] == cmd[y[sel]]).mean()), 1),
                        "cnn_sozinha_pct": round(100 * float((so_cnn[sel] == y[sel]).mean()), 1),
                        "quem_decide": "contagem de dedos" if c in NUMEROS_CARRO else "CNN 3D",
                        "fontes": sorted({s["fonte"] for s, ok in zip(te, sel) if ok}),
                        "erros_mais_comuns": dict(sorted(erros.items(), key=lambda t: -t[1])[:3])}
    r = {"o_que_e": __doc__.split("\n\n")[0],
         "criterio": "teste misto, pessoas fora do treino; por clipe; acerto = a classe prevista é a do clipe (36 classes possíveis)",
         "gestos_do_carro": bloco(NUMEROS_CARRO + MOVIMENTOS), "numeros": bloco(NUMEROS_CARRO), "movimentos": bloco(MOVIMENTOS),
         "todas_as_36_classes_pct": round(100 * float((pred == y).mean()), 1), "amostras_total": int(len(y)),
         "por_gesto": por_gesto}
    saida = RAIZ / "resultados/pontos/gestos_do_carro.json"
    saida.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("gestos_do_carro", "numeros", "movimentos"):
        print(k, r[k])
    print("36 classes:", r["todas_as_36_classes_pct"])
    for c, v in por_gesto.items():
        print(f"  {c:20s} n={v['amostras']:4d} acerto {v['acerto_pct']:5.1f}% (CNN sozinha {v['cnn_sozinha_pct']:5.1f}%) {v['fontes']} erros {v['erros_mais_comuns']}")


if __name__ == "__main__":
    main()
