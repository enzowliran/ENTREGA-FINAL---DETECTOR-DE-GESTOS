"""Compara o modelo com gestos novos ao de origem no fluxo simulado, com o gatilho de 0,7 e 3 janelas.

Só usa a validação do Jester. Mede os falsos disparos por minuto, inclusive os que vêm dos gestos novos,
e a detecção dos gestos antigos. Aplica os critérios fixados antes do treino, exceto o da webcam.

Uso:
    python src/avaliar_mais_gestos.py --modelo resultados/jester/cnn_3d_mais_gestos/melhor.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import calibrar_por_gesto as P
import jester as J
import modelos as M
import treinar_jester as T

MAX_FALSOS_POR_MIN = 0.1
MIN_TOP1_ANTIGOS = 0.923
NOVAS = ("One Finger", "Two Fingers", "Three Fingers", "Four Fingers", "Fist")


@torch.no_grad()
def acerto_antigos(caminho, dados, lote=48):
    """Acerto em toda a validação do Jester, com todas as saídas ativas: um gesto antigo lido como novo conta como erro."""
    estado = torch.load(caminho, map_location="cuda", weights_only=False)
    modelo = M.criar_modelo(estado["arquitetura"], len(estado["nomes"])).cuda().eval()
    modelo.load_state_dict(estado["modelo"])
    _, _, validacao, _ = J.ler_rotulos(Path(dados) / "rotulos/jester_labels")
    certos, total = 0, 0
    for clipes, classes in J.lotes(J.ConjuntoJester(dados, validacao, treino=False), lote, seed=0, embaralhar=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            previsto = modelo(T.preparar(clipes, "cuda")).float().argmax(1).cpu()
        certos += (previsto == classes).sum().item()
        total += len(classes)
    return certos / total


def medir(modelo, cache, dados, bloco):
    d = P.extrair_janelas(argparse.Namespace(modelo=str(modelo), cache=str(cache), dados=dados, bloco=bloco))
    nomes = [str(n) for n in d["nomes"]]
    gestos = [i for i in range(len(nomes)) if i not in d["negativos"].tolist()]
    limiares = np.full(len(nomes), np.inf)
    limiares[gestos] = P.LIMIAR_BASE
    _, classes = P.disparos(d["classe_neg"], d["conf_neg"], limiares, P.N_BASE)
    acerto, errados = P.avaliar_positivos(d, limiares)
    minutos = float(d["minutos"])
    novas = [nomes.index(n) for n in NOVAS if n in nomes]
    n_janelas, n_classes = P.disparos(d["classe_pos"], d["conf_pos"], limiares, P.N_BASE)
    return {"minutos": minutos, "falsos": len(classes), "falsos_por_min": len(classes) / minutos,
            "falsos_de_gestos_novos": int(np.isin(classes, novas).sum()),
            "falsos_por_classe": {nomes[int(c)]: int((classes == c).sum()) for c in np.unique(classes)},
            "deteccao_antigos": float(acerto.mean()), "classe_errada": float(errados.sum() / len(d["alvo"])),
            "gestos_antigos_disparando_novo": int(np.isin(n_classes, novas).sum()), "videos": len(d["alvo"])}


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--modelo", required=True)
    a.add_argument("--origem", default="resultados/jester/cnn_3d/melhor.pt")
    a.add_argument("--dados", default="data/Jester")
    a.add_argument("--bloco", type=int, default=1500)
    args = a.parse_args()
    pasta = Path(args.modelo).parent
    origem = medir(args.origem, "data/interno/janelas_validacao_cnn_3d.npz", args.dados, args.bloco)
    novo = medir(args.modelo, f"data/interno/janelas_validacao_{pasta.name}.npz", args.dados, args.bloco)
    top1 = acerto_antigos(args.modelo, args.dados)
    print(f"Acerto nos gestos antigos, validação completa: {top1:.4f} (critério: pelo menos {MIN_TOP1_ANTIGOS})", flush=True)
    print(f"{'Medida':32s} {'origem':>10s} {'novo':>10s}")
    for chave in ("falsos", "falsos_por_min", "falsos_de_gestos_novos", "deteccao_antigos", "classe_errada", "gestos_antigos_disparando_novo"):
        print(f"{chave:32s} {origem[chave]:10.4f} {novo[chave]:10.4f}")
    print("Falsos por classe no novo:", novo["falsos_por_classe"])
    resumo = {"origem": origem, "novo": novo, "top1_antigos_validacao": top1, "criterio_top1": MIN_TOP1_ANTIGOS, "passou_top1": top1 >= MIN_TOP1_ANTIGOS,
              "criterio_falsos_por_min": MAX_FALSOS_POR_MIN, "passou_falsos": novo["falsos_por_min"] <= MAX_FALSOS_POR_MIN}
    print("Critério de acerto nos gestos antigos:", "PASSOU" if resumo["passou_top1"] else "NÃO PASSOU")
    print("Critério de falsos por minuto:", "PASSOU" if resumo["passou_falsos"] else "NÃO PASSOU")
    (pasta / "fluxo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
