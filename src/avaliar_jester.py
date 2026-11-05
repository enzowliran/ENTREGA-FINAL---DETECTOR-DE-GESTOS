"""Mede falsas ativações de um modelo salvo: em vídeos sem gesto da validação e em ruído.

Uso:
    python src/avaliar_jester.py --modelo resultados/jester/cnn_3d_pequeno/melhor.pt --limite 5000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import jester as J
import modelos as M
import treinar_jester as T


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--modelo", required=True)
    a.add_argument("--dados", default="data/Jester")
    a.add_argument("--limite", type=int, default=5000, help="Vídeos da validação; os mesmos usados no treino pequeno")
    args = a.parse_args()
    estado = torch.load(args.modelo, map_location="cuda", weights_only=False)
    nomes = estado["nomes"]
    modelo = M.criar_modelo(estado["arquitetura"], len(nomes)).cuda()
    modelo.load_state_dict(estado["modelo"])
    negativos = torch.tensor([nomes.index(n) for n in T.SEM_GESTO], device="cuda")
    _, _, validacao, _ = J.ler_rotulos(Path(args.dados) / "rotulos/jester_labels")
    validacao = dict(list(validacao.items())[:args.limite])
    metricas = T.avaliar(modelo, J.ConjuntoJester(args.dados, validacao, treino=False), 48, "cuda", negativos)

    # Ruído e imagens sem movimento nunca vistos: nenhum deles deveria ativar um gesto.
    gerador = np.random.default_rng(0)
    x = {"ruido": gerador.uniform(0, 255, (256, 16, 3, 96, 96)).astype(np.float32),
         "cor_lisa": np.tile(gerador.uniform(0, 255, (256, 1, 1, 1, 1)).astype(np.float32), (1, 16, 3, 96, 96))}
    modelo.eval()
    sinteticos = {}
    for nome, dados in x.items():
        probabilidades = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(dados), 48):
                lote = torch.from_numpy(dados[i:i + 48]).cuda()
                probabilidades.append(modelo(T.preparar(lote, "cuda")).float().softmax(1))
        p = torch.cat(probabilidades)
        gesto = p.clone()
        gesto[:, negativos] = 0
        sinteticos[nome] = {str(l): float((gesto.max(1).values >= l).float().mean()) for l in T.LIMIARES}
    saida = {"top1": metricas["top1"], "top5": metricas["top5"], "n": metricas["n"], "curva": metricas["curva"], "sinteticos": sinteticos}
    destino = Path(args.modelo).with_name("avaliacao_falsas.json")
    destino.write_text(json.dumps(saida, indent=2), encoding="utf-8")
    print(f"top1 {saida['top1']:.4f} em {saida['n']} vídeos")
    print("limiar  falsa_ativacao  gestos_acertados  gestos_errados")
    for c in saida["curva"]:
        print(f"{c['limiar']:5.2f}  {c['falsa_ativacao']:14.4f}  {c['gestos_acertados']:16.4f}  {c['gestos_errados']:14.4f}")
    print("ativações em dados sintéticos:", {k: (v["0.6"], v["0.9"]) for k, v in sinteticos.items()}, "(limiar 0,6 e 0,9)")


if __name__ == "__main__":
    main()
