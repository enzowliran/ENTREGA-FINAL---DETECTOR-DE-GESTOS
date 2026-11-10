"""Compara dois modelos, classe por classe, nos mesmos vídeos de validação do Jester: o que caiu e para onde foi.

Uso:
    python src/comparar_jester_por_classe.py resultados/jester/cnn_3d_mais_gestos/melhor.pt resultados/jester/cnn_3d_dedos/ultimo.pt
(o segundo pode ser um ultimo.pt de treino; os nomes das classes vêm do primeiro)
"""
import random
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import jester as J
import modelos as M
from treinar_gestos_novos import prever


def carregar(caminho, nomes, arquitetura):
    e = torch.load(caminho, map_location="cuda", weights_only=False)
    m = M.criar_modelo(e.get("arquitetura", arquitetura), len(nomes)).cuda().eval()
    m.load_state_dict(e["modelo"])
    return m


def main():
    a, b = sys.argv[1], sys.argv[2]
    ea = torch.load(a, map_location="cpu", weights_only=False)
    nomes, arq = list(ea["nomes"]), ea["arquitetura"]
    _, _, validacao, _ = J.ler_rotulos(Path("data/Jester/rotulos/jester_labels"))
    itens = list(validacao.items())
    random.Random(42).shuffle(itens)  # o mesmo sorteio do treino
    conj = J.ConjuntoJester("data/Jester", dict(itens[:3000]), treino=False)
    pa, c = prever(carregar(a, nomes, arq), conj, 48)
    pb, _ = prever(carregar(b, nomes, arq), conj, 48)
    ya, yb = pa.argmax(1), pb.argmax(1)
    print(f"top1: {float((ya == c).float().mean()):.3f} -> {float((yb == c).float().mean()):.3f}")
    # "Sem gesto" e "Outro movimento" juntos: trocar um pelo outro não dispara comando nenhum.
    neg = torch.tensor([nomes.index("No gesture"), nomes.index("Doing other things")])
    junta = lambda y: torch.where(torch.isin(y, neg), neg[0], y)
    print(f"top1 com as duas classes negativas juntas: {float((junta(ya) == junta(c)).float().mean()):.3f} -> {float((junta(yb) == junta(c)).float().mean()):.3f}")
    gesto = ~torch.isin(c, neg)
    print(f"só nos vídeos com gesto: {float((ya[gesto] == c[gesto]).float().mean()):.3f} -> {float((yb[gesto] == c[gesto]).float().mean()):.3f}")
    print(f"sem gesto que vira gesto (disparo possível): {float((~torch.isin(ya[~gesto], neg)).float().mean()):.3f} -> {float((~torch.isin(yb[~gesto], neg)).float().mean()):.3f}")
    linhas = []
    for k in c.unique().tolist():
        s = c == k
        acc_a, acc_b = float((ya[s] == k).float().mean()), float((yb[s] == k).float().mean())
        destinos = Counter(nomes[int(x)] for x in yb[s & (yb != k)])
        linhas.append((acc_b - acc_a, nomes[k], int(s.sum()), acc_a, acc_b, destinos.most_common(3)))
    for d, n, q, x, y, dest in sorted(linhas)[:10]:
        print(f"{n:32s} n={q:4d}  {x:.2f} -> {y:.2f} ({d:+.2f})  erros vão para: {dest}")
    errados_b = Counter(nomes[int(x)] for x in yb[yb != c])
    errados_a = Counter(nomes[int(x)] for x in ya[ya != c])
    print("\nPara onde vão os erros (antes -> depois):")
    for n, q in errados_b.most_common(8):
        print(f"  {n:32s} {errados_a.get(n, 0):4d} -> {q:4d}")


if __name__ == "__main__":
    main()
