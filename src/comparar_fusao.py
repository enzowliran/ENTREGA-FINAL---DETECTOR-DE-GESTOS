"""Pontos + CNN: junta as saídas da GRU sobre pontos e da CNN 3D nas mesmas amostras de teste do teste misto.

Lê as probabilidades salvas por src/comparar_misto.py em data/pontos/misto_modelos/. Três formas de juntar:
  - média: p = w * p_GRU + (1 - w) * p_CNN;
  - portão pela mão: quando o MediaPipe achou a mão em >= 70% dos quadros, usa a média; senão, só a CNN;
  - o melhor dos dois por classe não é usado (seria escolher no próprio teste).
O peso w é escolhido sem olhar para as amostras medidas: o teste é dividido em duas metades (sorteio fixo, por base),
w é escolhido numa metade e medido na outra, e vice-versa. O resultado é a soma das duas metades medidas.

Uso:
    demo\\python.bat src\\comparar_fusao.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "data/pontos/misto_modelos"
PESOS = np.round(np.arange(0, 1.01, 0.1), 1)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--cnn", default="cnn", help="cnn (do zero) ou cnn_pre (partindo do Jester inteiro)")
    a.add_argument("--saida", default=None)
    args = a.parse_args()
    y = np.load(PASTA / "teste_y.npy")
    mao = np.load(PASTA / "teste_mao.npy") >= 0.7
    fonte = np.load(PASTA / "teste_fonte.npy")
    pg = np.load(PASTA / "gru_probs.npy")
    pc = np.load(PASTA / f"{args.cnn}_probs.npy")
    metade = np.zeros(len(y), bool)
    g = np.random.default_rng(0)
    for f in np.unique(fonte):
        i = np.nonzero(fonte == f)[0]
        metade[g.permutation(i)[:len(i) // 2]] = True

    def juntar(w, portao):
        p = w * pg + (1 - w) * pc
        if portao:
            p = np.where(mao[:, None], p, pc)
        return p.argmax(1)

    def cruzado(portao):
        """Escolhe w numa metade e mede na outra; devolve as previsões e os pesos escolhidos."""
        pred, pesos = np.zeros(len(y), int), []
        for escolha in (metade, ~metade):
            w = max(PESOS, key=lambda w: (juntar(w, portao)[escolha] == y[escolha]).mean())
            pesos.append(float(w))
            pred[~escolha] = juntar(w, portao)[~escolha]
        return pred, pesos

    preds = {"GRU sobre pontos": pg.argmax(1), "CNN 3D": pc.argmax(1)}
    preds["Pontos + CNN (média)"], w_media = cruzado(False)
    preds["Pontos + CNN (média só com mão)"], w_portao = cruzado(True)
    r = {"cnn": args.cnn, "pesos_escolhidos": {"média": w_media, "média só com mão": w_portao},
         "curva_pesos_todo_teste": {str(w): round(100 * (juntar(w, False) == y).mean(), 1) for w in PESOS}, "modelos": {}}
    for nome, p in preds.items():
        ok = p == y
        r["modelos"][nome] = {"todos": round(100 * ok.mean(), 1), "com_mao": round(100 * ok[mao].mean(), 1),
                              "por_fonte": {f: round(100 * ok[fonte == f].mean(), 1) for f in np.unique(fonte)}}
    destino = RAIZ / (args.saida or f"resultados/pontos/misto_fusao_{args.cnn}.json")
    destino.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    fontes = list(np.unique(fonte))
    print("modelo | todos | com mão | " + " | ".join(fontes))
    for nome, m in r["modelos"].items():
        print(f"{nome} | {m['todos']} | {m['com_mao']} | " + " | ".join(str(m['por_fonte'][f]) for f in fontes))
    print(f"pesos da GRU escolhidos em cada metade: média {w_media}, só com mão {w_portao}")


if __name__ == "__main__":
    main()
