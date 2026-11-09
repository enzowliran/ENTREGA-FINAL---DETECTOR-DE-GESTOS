"""Gera as matrizes de confusão usadas na apresentação, só com dados de validação.

- cnn_3d na validação do Jester (14.787 vídeos, 27 classes);
- cnn_3d_mais_gestos nas fotos de validação do HaGRID (gestos novos e os equivalentes antigos).

O teste não é usado. As contagens ficam em JSON, ao lado de cada modelo.

Uso:
    python src/matriz_confusao.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import jester as J
import modelos as M
import treinar_gestos_novos as N


def carregar(caminho):
    estado = torch.load(caminho, map_location="cuda", weights_only=False)
    modelo = M.criar_modelo(estado["arquitetura"], len(estado["nomes"])).cuda().eval()
    modelo.load_state_dict(estado["modelo"])
    return modelo, list(estado["nomes"])


def contar(p, c, n):
    m = np.zeros((n, n), dtype=np.int64)
    np.add.at(m, (c.numpy(), p.argmax(1).numpy()), 1)
    return m


def main():
    modelo, nomes = carregar("resultados/jester/cnn_3d/melhor.pt")
    _, _, validacao, _ = J.ler_rotulos(Path("data/Jester/rotulos/jester_labels"))
    p, c = N.prever(modelo, J.ConjuntoJester("data/Jester", validacao, treino=False), 48)
    m = contar(p, c, len(nomes))
    saida = {"nomes": nomes, "contagens": m.tolist(), "n": int(m.sum()), "acerto": float(np.trace(m) / m.sum()), "conjunto": "validação do Jester"}
    Path("resultados/jester/cnn_3d/matriz_confusao_validacao.json").write_text(json.dumps(saida, ensure_ascii=False), encoding="utf-8")
    print(f"Jester: {saida['n']} vídeos, acerto {saida['acerto']:.4f}", flush=True)

    modelo, nomes = carregar("resultados/jester/cnn_3d_mais_gestos/melhor.pt")
    anotacoes = json.loads(Path("data/HaGRID/anotacoes.json").read_text(encoding="utf-8"))
    conjunto = N.ConjuntoHagrid("data/HaGRID", "val", anotacoes, nomes, treino=False)
    p, c = N.prever(modelo, conjunto, 48)
    m = contar(p, c, len(nomes))
    saida = {"nomes": nomes, "contagens": m.tolist(), "n": int(m.sum()), "acerto": float(np.trace(m) / m.sum()), "conjunto": "fotos de validação do HaGRID"}
    Path("resultados/jester/cnn_3d_mais_gestos/matriz_confusao_hagrid_val.json").write_text(json.dumps(saida, ensure_ascii=False), encoding="utf-8")
    print(f"HaGRID: {saida['n']} fotos, acerto {saida['acerto']:.4f}", flush=True)


if __name__ == "__main__":
    main()
