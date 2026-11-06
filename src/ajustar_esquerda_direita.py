"""Ajuste fino curto do modelo do Jester, mostrando mais vezes deslizar para a esquerda e a direita.

Parte dos pesos treinados aqui mesmo, com taxa de aprendizado baixa, e nunca altera o modelo de
origem. A época escolhida é a de maior acerto médio em esquerda e direita na validação, entre as
que não perderam mais de 0,2 ponto no acerto geral. Depois mede o fluxo simulado com o gatilho da
demo. O conjunto de teste não é usado.

Uso:
    python src/ajustar_esquerda_direita.py
"""
import argparse
import csv
import ctypes
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import calibrar_por_gesto as P
import jester as J
import modelos as M
import treinar_jester as T

ESQUERDA, DIREITA = "Swiping Left", "Swiping Right"
MARGEM_TOP1 = 0.002


@torch.no_grad()
def avaliar(modelo, conjunto, lote, nomes):
    modelo.eval()
    previstos, classes_todas = [], []
    for clipes, classes in J.lotes(conjunto, lote, seed=0, embaralhar=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            previstos.append(modelo(T.preparar(clipes, "cuda")).float().argmax(1).cpu())
        classes_todas.append(classes)
    p, c = torch.cat(previstos).numpy(), torch.cat(classes_todas).numpy()
    e, d = nomes.index(ESQUERDA), nomes.index(DIREITA)
    return {"top1": float((p == c).mean()), "esquerda": float((p[c == e] == e).mean()), "direita": float((p[c == d] == d).mean()),
            "esq_vira_dir": float((p[c == e] == d).mean()), "dir_vira_esq": float((p[c == d] == e).mean())}


def medir_fluxo(modelo_pt, cache, dados, bloco):
    """Janelas do fluxo simulado com o gatilho de 0,7 e 3 janelas, inclusive a troca entre os lados."""
    d = P.extrair_janelas(argparse.Namespace(modelo=str(modelo_pt), cache=str(cache), dados=dados, bloco=bloco))
    nomes = [str(n) for n in d["nomes"]]
    gestos = [i for i in range(len(nomes)) if i not in d["negativos"].tolist()]
    limiares = np.full(len(nomes), np.inf)
    limiares[gestos] = P.LIMIAR_BASE
    falsos, classes_falsos = P.disparos(d["classe_neg"], d["conf_neg"], limiares, P.N_BASE)
    acerto, errados = P.avaliar_positivos(d, limiares)
    janelas, classes = P.disparos(d["classe_pos"], d["conf_pos"], limiares, P.N_BASE)
    videos = d["video_pos"][janelas]
    e, c = nomes.index(ESQUERDA), nomes.index(DIREITA)
    me, md = d["alvo"] == e, d["alvo"] == c
    n = len(d["alvo"])
    return {"falsos": len(falsos), "falsos_por_min": len(falsos) / float(d["minutos"]),
            "falsos_esquerda_direita": int(np.isin(classes_falsos, [e, c]).sum()), "deteccao": float(acerto.mean()),
            "classe_errada": float(errados.sum() / n), "det_esquerda": float(acerto[me].mean()), "det_direita": float(acerto[md].mean()),
            "esq_vira_dir": float((classes[d["alvo"][videos] == e] == c).sum() / me.sum()),
            "dir_vira_esq": float((classes[d["alvo"][videos] == c] == e).sum() / md.sum())}


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--origem", default="resultados/jester/cnn_3d/melhor.pt")
    a.add_argument("--dados", default="data/Jester")
    a.add_argument("--nome", default="cnn_3d_esq_dir")
    a.add_argument("--epocas", type=int, default=4)
    a.add_argument("--repeticoes", type=int, default=2, help="Vezes que cada vídeo de esquerda e direita aparece por época")
    a.add_argument("--outros", type=int, default=20000, help="Vídeos das outras classes sorteados por época")
    a.add_argument("--lr", type=float, default=1e-4)
    a.add_argument("--lote", type=int, default=48)
    a.add_argument("--bloco", type=int, default=1500)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--sem-fluxo", action="store_true", help="Teste rápido: não mede o fluxo")
    args = a.parse_args()

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    origem = Path(args.origem)
    pasta = Path("resultados/jester") / args.nome
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    config = json.loads((origem.parent / "config.json").read_text(encoding="utf-8"))
    estado = torch.load(origem, map_location="cuda", weights_only=False)
    nomes = estado["nomes"]
    modelo = M.criar_modelo(estado["arquitetura"], len(nomes)).cuda()
    modelo.load_state_dict(estado["modelo"])

    _, treino, validacao, _ = J.ler_rotulos(Path(args.dados) / "rotulos/jester_labels")
    c_treino = J.ConjuntoJester(args.dados, treino, treino=True, frac_sinteticos=config.get("frac_sinteticos", 0.0),
                                classe_sem_gesto=nomes.index("No gesture"),
                                classes_estaticas=[nomes.index(n) for n in T.ESTATICAS + T.SEM_GESTO],
                                aug_tempo=config.get("aug_tempo", False), aug_espaco=config.get("aug_espaco", False))
    c_val = J.ConjuntoJester(args.dados, validacao, treino=False)
    alvos = {nomes.index(ESQUERDA), nomes.index(DIREITA)}
    negativos = torch.tensor([nomes.index(n) for n in T.SEM_GESTO], device="cuda")
    todas = c_treino.amostras
    reforco = [s for s in todas if s[3] in alvos]
    outras = [s for s in todas if s[3] not in alvos]
    print(f"Por época: {len(reforco)} vídeos de esquerda e direita x{args.repeticoes} + {args.outros} de outras classes", flush=True)

    base = avaliar(modelo, c_val, args.lote, nomes)
    print(f"Modelo de origem: top1 {base['top1']:.4f}; esquerda {base['esquerda']:.4f}; direita {base['direita']:.4f}; "
          f"esq->dir {base['esq_vira_dir']:.4f}; dir->esq {base['dir_vira_esq']:.4f}", flush=True)

    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=args.epocas)
    gerador = np.random.default_rng(args.seed)
    historico, escolhido, melhor_lados = [], None, -1.0
    for epoca in range(args.epocas):
        sorteio = gerador.choice(len(outras), min(args.outros, len(outras)), replace=False)
        c_treino.amostras = reforco * args.repeticoes + [outras[int(i)] for i in sorteio]
        modelo.train()
        inicio, soma, total = time.time(), 0.0, 0
        for clipes, classes in J.lotes(c_treino, args.lote, seed=args.seed + epoca):
            classes = classes.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                saida = modelo(T.preparar(clipes, "cuda"))
                perda = nn.functional.cross_entropy(saida.float(), classes, label_smoothing=config.get("suavizacao", 0.1))
                perda = perda + config.get("peso_falso", 0.0) * T.perda_falsa_ativacao(saida.float(), classes, negativos)
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            otimizador.step()
            soma += perda.item() * classes.numel(); total += classes.numel()
        agendador.step()
        m = avaliar(modelo, c_val, args.lote, nomes)
        lados = (m["esquerda"] + m["direita"]) / 2
        linha = {"epoca": epoca, "perda_treino": soma / total, **m, "minutos": (time.time() - inicio) / 60}
        historico.append(linha)
        print(f"Época {epoca + 1}/{args.epocas}: top1 {m['top1']:.4f}; esquerda {m['esquerda']:.4f}; direita {m['direita']:.4f}; "
              f"esq->dir {m['esq_vira_dir']:.4f}; dir->esq {m['dir_vira_esq']:.4f}; {linha['minutos']:.1f} min", flush=True)
        if m["top1"] >= base["top1"] - MARGEM_TOP1 and lados > melhor_lados:
            melhor_lados, escolhido = lados, epoca
            torch.save({"modelo": modelo.state_dict(), "nomes": nomes, "epoca": epoca, "top1_val": m["top1"],
                        "arquitetura": estado["arquitetura"], "origem": str(origem)}, pasta / "melhor.pt")
    with (pasta / "historico.csv").open("w", newline="", encoding="utf-8") as f:
        e = csv.DictWriter(f, fieldnames=list(historico[0]))
        e.writeheader(); e.writerows(historico)

    resumo = {"origem_validacao": base, "historico": historico, "epoca_escolhida": escolhido}
    if escolhido is None:
        print("Nenhuma época manteve o acerto geral: fica o modelo de origem.", flush=True)
    elif not args.sem_fluxo:
        print(f"Época escolhida: {escolhido + 1}. Medindo o fluxo...", flush=True)
        antes = medir_fluxo(origem, "data/interno/janelas_validacao_cnn_3d.npz", args.dados, args.bloco)
        depois = medir_fluxo(pasta / "melhor.pt", f"data/interno/janelas_validacao_{args.nome}.npz", args.dados, args.bloco)
        resumo.update(fluxo_origem=antes, fluxo_ajustado=depois)
        print("\nFluxo com limiar 0,7 e 3 janelas (origem -> ajustado)")
        for chave in antes:
            print(f"  {chave:24s} {antes[chave]:.4f} -> {depois[chave]:.4f}")
    (pasta / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
