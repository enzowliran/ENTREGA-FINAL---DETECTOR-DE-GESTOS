"""Treina do zero uma rede no Jester e salva métricas e pesos.

O conjunto de teste só é avaliado uma vez, ao final, com o melhor modelo da validação.

Uso:
    python src/treinar_jester.py --modelo cnn_3d --epocas 20
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
import jester as J
import modelos as M

MEDIA, ESCALA = 127.5, 127.5
SEM_GESTO = ("No gesture", "Doing other things")
ESTATICAS = ("Thumb Up", "Thumb Down", "Stop Sign")
LIMIARES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def perda_falsa_ativacao(logits, classes, indices_negativos):
    """Custo extra só nos vídeos sem gesto: -log P(qualquer classe sem gesto)."""
    negativos = torch.isin(classes, indices_negativos)
    if not negativos.any():
        return logits.sum() * 0.0
    log_p = logits[negativos].log_softmax(1)[:, indices_negativos]
    return -torch.logsumexp(log_p, dim=1).sum() / len(classes)


def medir_ativacoes(p, classes, indices_negativos):
    """Falsa ativação nos vídeos sem gesto e acerto nos gestos, para cada limiar."""
    negativos = torch.isin(classes, indices_negativos.cpu())
    gesto = p.clone()
    gesto[:, indices_negativos.cpu()] = 0
    confianca, classe = gesto.max(1)
    return [{"limiar": l, "falsa_ativacao": float((confianca[negativos] >= l).float().mean()),
             "gestos_acertados": float(((confianca[~negativos] >= l) & (classe[~negativos] == classes[~negativos])).float().mean()),
             "gestos_errados": float(((confianca[~negativos] >= l) & (classe[~negativos] != classes[~negativos])).float().mean())}
            for l in LIMIARES]


def preparar(clipes, dispositivo):
    return (clipes.to(dispositivo, non_blocking=True) - MEDIA) / ESCALA


@torch.no_grad()
def avaliar(modelo, conjunto, tamanho_lote, dispositivo, indices_negativos):
    modelo.eval()
    perda, acertos, acertos5, total = 0.0, 0, 0, 0
    todas_p, todas_c = [], []
    for clipes, classes in J.lotes(conjunto, tamanho_lote, seed=0, embaralhar=False):
        classes = classes.to(dispositivo)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            saida = modelo(preparar(clipes, dispositivo)).float()
        perda += nn.functional.cross_entropy(saida, classes, reduction="sum").item()
        acertos += (saida.argmax(1) == classes).sum().item()
        acertos5 += saida.topk(5, 1).indices.eq(classes[:, None]).any(1).sum().item()
        total += classes.numel()
        todas_p.append(saida.softmax(1).cpu()); todas_c.append(classes.cpu())
    curva = medir_ativacoes(torch.cat(todas_p), torch.cat(todas_c), indices_negativos)
    return {"perda": perda / total, "top1": acertos / total, "top5": acertos5 / total, "n": total, "curva": curva}


def main():
    analisador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analisador.add_argument("--modelo", required=True, choices=["cnn_gru", "cnn_transformer", "cnn_3d"])
    analisador.add_argument("--dados", default="data/Jester")
    analisador.add_argument("--epocas", type=int, default=20)
    analisador.add_argument("--lote", type=int, default=48)
    analisador.add_argument("--lr", type=float, default=1e-3)
    analisador.add_argument("--seed", type=int, default=42)
    analisador.add_argument("--limite-treino", type=int, help="Usa só N vídeos de treino (teste rápido)")
    analisador.add_argument("--retomar", action="store_true")
    analisador.add_argument("--peso-falso", type=float, default=0.0, help="Peso do custo extra de falsa ativação")
    analisador.add_argument("--aug-tempo", action="store_true", help="Treina com trechos e velocidades variados")
    analisador.add_argument("--aug-espaco", action="store_true", help="Treina com a mão menor e fora do centro")
    analisador.add_argument("--suavizacao", type=float, default=0.1, help="Suavização de rótulos; valores altos limitam a confiança")
    analisador.add_argument("--frac-sinteticos", type=float, default=0.0, help="Fração de negativos sintéticos no treino")
    analisador.add_argument("--nome", help="Pasta dos resultados; por padrão, o nome do modelo")
    args = analisador.parse_args()

    # Evita a suspensão por inatividade só enquanto o treino roda, sem mudar as configurações.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dispositivo = "cuda"
    pasta = Path("resultados/jester") / (args.nome or args.modelo)
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    nomes, treino, validacao, teste = J.ler_rotulos(Path(args.dados) / "rotulos/jester_labels")
    if args.limite_treino:
        treino = dict(list(treino.items())[:args.limite_treino])
        validacao = dict(list(validacao.items())[:args.limite_treino // 4])
    negativos = torch.tensor([nomes.index(n) for n in SEM_GESTO], device=dispositivo)
    c_treino = J.ConjuntoJester(args.dados, treino, treino=True, frac_sinteticos=args.frac_sinteticos,
                                classe_sem_gesto=nomes.index("No gesture"),
                                classes_estaticas=[nomes.index(n) for n in ESTATICAS + SEM_GESTO],
                                aug_tempo=args.aug_tempo, aug_espaco=args.aug_espaco)
    c_val = J.ConjuntoJester(args.dados, validacao, treino=False)
    print(f"Treino {len(c_treino)} (faltando {c_treino.faltando}); validação {len(c_val)} (faltando {c_val.faltando}); "
          f"{len(nomes)} classes", flush=True)

    modelo = M.criar_modelo(args.modelo, len(nomes)).to(dispositivo)
    print(f"{args.modelo}: {M.contar_parametros(modelo):,} parâmetros, inicialização aleatória", flush=True)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    passos = args.epocas * (len(c_treino) // args.lote + 1)
    agendador = torch.optim.lr_scheduler.OneCycleLR(otimizador, max_lr=args.lr, total_steps=passos, pct_start=0.1)

    historico = pasta / "historico.csv"
    melhor, inicio_epoca = -1.0, 0
    ultimo = pasta / "ultimo.pt"
    if args.retomar and ultimo.exists():
        estado = torch.load(ultimo, map_location=dispositivo)
        modelo.load_state_dict(estado["modelo"]); otimizador.load_state_dict(estado["otimizador"])
        agendador.load_state_dict(estado["agendador"])
        melhor, inicio_epoca = estado["melhor"], estado["epoca"] + 1
        print(f"Retomando da época {inicio_epoca}", flush=True)
    elif historico.exists():
        historico.unlink()

    for epoca in range(inicio_epoca, args.epocas):
        modelo.train()
        inicio = time.time()
        soma, certos, total = 0.0, 0, 0
        for passo, (clipes, classes) in enumerate(J.lotes(c_treino, args.lote, seed=args.seed + epoca)):
            classes = classes.to(dispositivo, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                saida = modelo(preparar(clipes, dispositivo))
                perda = nn.functional.cross_entropy(saida.float(), classes, label_smoothing=args.suavizacao)
                if args.peso_falso:
                    perda = perda + args.peso_falso * perda_falsa_ativacao(saida.float(), classes, negativos)
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            otimizador.step()
            if agendador.last_epoch < passos - 1:
                agendador.step()
            soma += perda.item() * classes.numel()
            certos += (saida.argmax(1) == classes).sum().item()
            total += classes.numel()
            if passo % 200 == 0:
                print(f" ep{epoca} passo {passo} perda {soma / total:.3f} top1 {certos / total:.3f} "
                      f"{total / (time.time() - inicio):.0f} vídeos/s GPU {torch.cuda.max_memory_allocated() >> 20} MB", flush=True)
        duracao = time.time() - inicio
        metricas = avaliar(modelo, c_val, args.lote, dispositivo, negativos)
        no_limiar = next(c for c in metricas["curva"] if c["limiar"] == 0.6)
        linha = {"epoca": epoca, "perda_treino": soma / total, "top1_treino": certos / total, "perda_val": metricas["perda"],
                 "top1_val": metricas["top1"], "top5_val": metricas["top5"], "falsa_ativacao_0.6": no_limiar["falsa_ativacao"],
                 "gestos_acertados_0.6": no_limiar["gestos_acertados"], "segundos": duracao}
        novo = not historico.exists()
        with historico.open("a", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=list(linha))
            if novo:
                escritor.writeheader()
            escritor.writerow(linha)
        print(f"Época {epoca}: treino {certos / total:.3f}, validação top1 {metricas['top1']:.4f} top5 {metricas['top5']:.4f}, "
              f"falsa ativação {no_limiar['falsa_ativacao']:.4f}, gestos acertados {no_limiar['gestos_acertados']:.4f} (limiar 0,6), "
              f"{duracao / 60:.1f} min", flush=True)
        (pasta / "curva_val.json").write_text(json.dumps(metricas["curva"], indent=2), encoding="utf-8")
        if metricas["top1"] > melhor:
            melhor = metricas["top1"]
            torch.save({"modelo": modelo.state_dict(), "nomes": nomes, "epoca": epoca, "top1_val": melhor,
                        "arquitetura": args.modelo}, pasta / "melhor.pt")
        torch.save({"modelo": modelo.state_dict(), "otimizador": otimizador.state_dict(), "agendador": agendador.state_dict(),
                    "epoca": epoca, "melhor": melhor}, ultimo)

    if args.limite_treino:
        print("Teste rápido: conjunto de teste não avaliado.", flush=True)
        return
    modelo.load_state_dict(torch.load(pasta / "melhor.pt", map_location=dispositivo)["modelo"])
    c_teste = J.ConjuntoJester(args.dados, teste, treino=False)
    resultado = avaliar(modelo, c_teste, args.lote, dispositivo, negativos)
    resultado["melhor_top1_val"] = melhor
    (pasta / "teste.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(f"Teste: top1 {resultado['top1']:.4f}, top5 {resultado['top5']:.4f} em {resultado['n']} vídeos", flush=True)


if __name__ == "__main__":
    main()
