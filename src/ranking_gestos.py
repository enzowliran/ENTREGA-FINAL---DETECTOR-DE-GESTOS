"""Ordena os gestos pelo desempenho no fluxo simulado da validação, com o gatilho de 0,7 e 3 janelas.

Usa as janelas salvas por calibrar_por_gesto.py, sem passar o modelo de novo nem usar o teste.
Para cada gesto: quantos vídeos dele foram detectados, quantos disparos dele foram corretos, com
qual gesto ele mais é confundido e quantos falsos disparos gerou nos vídeos sem gesto.

Uso:
    python src/ranking_gestos.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "demo"))
import calibrar_por_gesto as P
from ao_vivo import TRADUCAO


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--cache", default="data/interno/janelas_validacao_cnn_3d.npz")
    a.add_argument("--saida", default="resultados/jester/cnn_3d/ranking_gestos.csv")
    args = a.parse_args()
    d = dict(np.load(args.cache, allow_pickle=False))
    nomes = [str(n) for n in d["nomes"]]
    gestos = [i for i in range(len(nomes)) if i not in d["negativos"].tolist()]
    limiares = np.full(len(nomes), np.inf)
    limiares[gestos] = P.LIMIAR_BASE
    falsos, classes_falsos = P.disparos(d["classe_neg"], d["conf_neg"], limiares, P.N_BASE)
    janelas, classes = P.disparos(d["classe_pos"], d["conf_pos"], limiares, P.N_BASE)
    alvos = d["alvo"][d["video_pos"][janelas]]
    acerto, _ = P.avaliar_positivos(d, limiares)

    linhas = []
    for c in gestos:
        meus = alvos == c
        errados = classes[meus & (classes != c)]
        confusao = "-"
        if len(errados):
            valores, contagens = np.unique(errados, return_counts=True)
            confusao = TRADUCAO.get(nomes[int(valores[contagens.argmax()])], nomes[int(valores[contagens.argmax()])])
        disparos_dele = classes == c
        linhas.append({"gesto": TRADUCAO.get(nomes[c], nomes[c]), "videos": int((d["alvo"] == c).sum()),
                       "detectados": float(acerto[d["alvo"] == c].mean()),
                       "disparos_corretos": float((alvos[disparos_dele] == c).mean()) if disparos_dele.any() else 0.0,
                       "mais_confundido_com": confusao, "falsos_sem_gesto": int((classes_falsos == c).sum())})
    linhas.sort(key=lambda l: -l["detectados"])
    with open(args.saida, "w", newline="", encoding="utf-8") as f:
        e = csv.DictWriter(f, fieldnames=list(linhas[0]))
        e.writeheader(); e.writerows(linhas)
    for i, l in enumerate(linhas, 1):
        print(f"{i:2d} {l['gesto']:38s} det {l['detectados']:.1%} corretos {l['disparos_corretos']:.1%} "
              f"falsos {l['falsos_sem_gesto']} confunde {l['mais_confundido_com']}")


if __name__ == "__main__":
    main()
