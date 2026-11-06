"""Calcula ROC e AUC do modelo do Jester, por vídeo isolado, na validação e no teste.

Apenas mede o modelo salvo: nada aqui escolhe pesos, limiares ou configurações. Há duas leituras:
    - gesto x sem gesto: a nota é a maior probabilidade entre os 25 gestos, como no gatilho;
    - cada classe contra as outras, com a média simples das 27 classes (macro).

Uso:
    python src/avaliar_roc.py --modelo resultados/jester/cnn_3d/melhor.pt
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
import jester as J
import modelos as M
import treinar_jester as T


@torch.no_grad()
def prever(modelo, conjunto, lote):
    probs, classes = [], []
    for clipes, c in J.lotes(conjunto, lote, seed=0, embaralhar=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs.append(modelo(T.preparar(clipes, "cuda")).float().softmax(1).cpu())
        classes.append(c)
    return torch.cat(probs).numpy(), torch.cat(classes).numpy()


def medir(p, c, nomes):
    negativos = [nomes.index(n) for n in T.SEM_GESTO]
    gestos = [i for i in range(len(nomes)) if i not in negativos]
    e_gesto = ~np.isin(c, negativos)
    nota = p[:, gestos].max(1)
    fpr, tpr, limiares = roc_curve(e_gesto, nota)
    por_classe = {nomes[i]: float(roc_auc_score(c == i, p[:, i])) for i in range(len(nomes))}
    resumo = {"n": int(len(c)), "n_sem_gesto": int((~e_gesto).sum()), "acuracia": float((p.argmax(1) == c).mean()),
              "auc_gesto_x_sem_gesto": float(roc_auc_score(e_gesto, nota)), "auc_macro_27_classes": float(np.mean(list(por_classe.values()))),
              "auc_por_classe": por_classe,
              "tpr_com_fpr": {str(f): float(tpr[np.searchsorted(fpr, f, side="right") - 1]) for f in (0.001, 0.01, 0.05, 0.1)}}
    return resumo, (fpr, tpr, limiares)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--modelo", required=True)
    a.add_argument("--dados", default="data/Jester")
    a.add_argument("--lote", type=int, default=48)
    args = a.parse_args()
    estado = torch.load(args.modelo, map_location="cuda", weights_only=False)
    nomes = estado["nomes"]
    modelo = M.criar_modelo(estado["arquitetura"], len(nomes)).cuda().eval()
    modelo.load_state_dict(estado["modelo"])
    _, _, validacao, teste = J.ler_rotulos(Path(args.dados) / "rotulos/jester_labels")
    pasta = Path(args.modelo).parent
    saida = {}
    for nome, rotulos in (("validacao", validacao), ("teste", teste)):
        p, c = prever(modelo, J.ConjuntoJester(args.dados, rotulos, treino=False), args.lote)
        saida[nome], (fpr, tpr, limiares) = medir(p, c, nomes)
        with (pasta / f"roc_gesto_{nome}.csv").open("w", newline="", encoding="utf-8") as f:
            e = csv.writer(f)
            e.writerow(["falso_positivo", "verdadeiro_positivo", "limiar"])
            e.writerows(zip(fpr.round(6), tpr.round(6), limiares.round(6)))
        r = saida[nome]
        ordem = sorted(r["auc_por_classe"].items(), key=lambda x: x[1])
        print(f"{nome}: n={r['n']} sem gesto={r['n_sem_gesto']} acurácia={r['acuracia']:.4f} "
              f"AUC gesto x sem gesto={r['auc_gesto_x_sem_gesto']:.4f} AUC macro={r['auc_macro_27_classes']:.4f}", flush=True)
        print("  detecção com taxa de falso positivo:", {k: round(v, 4) for k, v in r["tpr_com_fpr"].items()}, flush=True)
        print("  menores AUC:", [(n, round(v, 4)) for n, v in ordem[:4]], flush=True)
        print("  maiores AUC:", [(n, round(v, 4)) for n, v in ordem[-3:]], flush=True)
    (pasta / "roc_auc.json").write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
