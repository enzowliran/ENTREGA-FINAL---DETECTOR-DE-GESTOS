"""Treina uma rede no DriverMHG, do zero ou a partir do nosso pré-treino no Jester.

Divisão: pessoas 19-25 ficam no teste oficial. As pessoas 16-18 viram validação, usada para
escolher a época; o teste não participa dessa escolha. As predições vão para CSV, para a calibração
de limiares e as curvas saírem dos arquivos.

Uso:
    python src/treinar_drivermhg.py --modelo cnn_3d --mao direita --modalidade ir --inicio zero
    python src/treinar_drivermhg.py --modelo cnn_3d --mao direita --modalidade ir --inicio resultados/jester/cnn_3d/melhor.pt
"""
import argparse
import csv
import ctypes
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import drivermhg as D
import modelos as M

CLASSES = ["FlickDown", "FlickUp", "SwipeLeft", "SwipeRight", "Tap", "None", "Other"]
PESSOAS_VAL = (16, 17, 18)
PASTA_ANOTACAO = "anotacao/media/kop/760E6E913CA0F70B/PC_backup/datasets/DriverMHG/Annotation"
CANAIS = {"rgb": 3, "ir": 1, "rgb_ir": 4}


class ConjuntoDriver:
    def __init__(self, tabela, prep, modalidade, n_frames=16, treino=False):
        self.tabela, self.prep, self.modalidade = tabela.reset_index(drop=True), Path(prep), modalidade
        self.n_frames, self.treino, self.abertos = n_frames, treino, {}

    def __len__(self):
        return len(self.tabela)

    def _abrir(self, pessoa, gravacao, tipo):
        chave = (pessoa, gravacao, tipo)
        if chave not in self.abertos:
            self.abertos[chave] = np.load(self.prep / f"S{pessoa:02d}_R{gravacao}_{tipo}.npy", mmap_mode="r")
        return self.abertos[chave]

    def carregar(self, posicao, gerador):
        s = self.tabela.iloc[posicao]
        inicio, fim = int(s.inicio), max(int(s.fim), int(s.inicio) + 1)
        pontos = np.linspace(inicio, fim, self.n_frames + 1)
        if self.treino:
            indices = (pontos[:-1] + gerador.random(self.n_frames) * (pontos[1:] - pontos[:-1])).astype(int)
        else:
            indices = ((pontos[:-1] + pontos[1:]) / 2).astype(int)
        partes = []
        if self.modalidade in ("rgb", "rgb_ir"):
            rgb = self._abrir(int(s.pessoa), int(s.gravacao), "rgb")
            partes.append(np.asarray(rgb[np.minimum(indices, len(rgb) - 1)], dtype=np.float32))
        if self.modalidade in ("ir", "rgb_ir"):
            ir = self._abrir(int(s.pessoa), int(s.gravacao), "ir")
            partes.append(np.asarray(ir[np.minimum(indices, len(ir) - 1)], dtype=np.float32)[..., None])
        clipe = np.concatenate(partes, axis=-1)
        if self.treino:
            # Sem inversão horizontal: ela trocaria a classe, como avisa o README oficial.
            altura, largura = clipe.shape[1:3]
            fator = gerador.uniform(0.85, 1.0)
            a, l = int(altura * fator), int(largura * fator)
            y0, x0 = gerador.integers(0, altura - a + 1), gerador.integers(0, largura - l + 1)
            clipe = np.stack([cv2.resize(q[y0:y0 + a, x0:x0 + l], (largura, altura)).reshape(altura, largura, -1) for q in clipe])
            clipe = clipe * gerador.uniform(0.8, 1.2)
        return clipe.transpose(0, 3, 1, 2), int(s.alvo)


def lotes(conjunto, tamanho, seed, executor, embaralhar):
    ordem = np.random.default_rng(seed).permutation(len(conjunto)) if embaralhar else np.arange(len(conjunto))
    for i in range(0, len(ordem), tamanho):
        parte = ordem[i:i + tamanho]
        if embaralhar and len(parte) < 2:
            continue
        r = list(executor.map(lambda p: conjunto.carregar(int(p), np.random.default_rng(seed * 100_003 + int(p))), parte))
        yield torch.from_numpy(np.stack([x[0] for x in r])), torch.tensor([x[1] for x in r]), parte


@torch.no_grad()
def prever(modelo, conjunto, tamanho, executor):
    modelo.eval()
    probabilidades, alvos, indices = [], [], []
    for clipes, classes, parte in lotes(conjunto, tamanho, 0, executor, False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            saida = modelo((clipes.cuda(non_blocking=True) - 127.5) / 127.5).float()
        probabilidades.append(saida.softmax(1).cpu().numpy())
        alvos.append(classes.numpy()); indices.append(parte)
    return np.concatenate(probabilidades), np.concatenate(alvos), np.concatenate(indices)


def salvar_predicoes(caminho, conjunto, p, alvos, indices):
    tabela = conjunto.tabela.iloc[indices][["pessoa", "gravacao", "mao", "gesto", "inicio", "fim"]].copy()
    tabela["alvo"] = alvos
    for j, nome in enumerate(CLASSES):
        tabela[f"p_{nome}"] = p[:, j].round(5)
    tabela.to_csv(caminho, index=False)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--modelo", required=True, choices=["cnn_gru", "cnn_transformer", "cnn_3d"])
    a.add_argument("--mao", required=True, choices=["direita", "esquerda"])
    a.add_argument("--modalidade", default="ir", choices=list(CANAIS))
    a.add_argument("--inicio", default="zero", help="'zero' ou o caminho de um melhor.pt do Jester")
    a.add_argument("--base", default="data/DriverMHG")
    a.add_argument("--pessoas-treino", type=int, default=15, help="Quantas das pessoas 1-15 usar no treino")
    a.add_argument("--epocas", type=int, default=40)
    a.add_argument("--congelar", type=int, default=3, help="Épocas treinando só a cabeça, se houver pré-treino")
    a.add_argument("--lote", type=int, default=24)
    a.add_argument("--lr", type=float, default=None)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--nome", default=None)
    a.add_argument("--pessoas-teste", type=int, nargs="+", help="Troca o teste oficial por estas pessoas (só no treino pequeno)")
    a.add_argument("--so-disponiveis", action="store_true", help="Usa só gravações já preparadas (teste de fluxo)")
    args = a.parse_args()

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    base = Path(args.base); prep = base / "prep"
    pretreino = args.inicio != "zero"
    nome = args.nome or f"{args.modelo}_{args.mao}_{args.modalidade}_{'jester' if pretreino else 'zero'}_p{args.pessoas_treino}"
    pasta = Path("resultados/drivermhg") / nome
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    tabela = D.ler_listas(base / PASTA_ANOTACAO)
    tabela = tabela[tabela.mao == args.mao].copy()
    tabela["alvo"] = tabela.gesto.map({n: i for i, n in enumerate(CLASSES)})
    if args.so_disponiveis:
        tem = tabela.apply(lambda s: (prep / f"S{s.pessoa:02d}_R{s.gravacao}_ir.npy").exists(), axis=1)
        tabela = tabela[tem]
    pessoas_teste = args.pessoas_teste or list(D.PESSOAS_TESTE)
    pessoas_val = () if args.pessoas_teste else PESSOAS_VAL
    pessoas_treino = sorted(set(tabela.pessoa) - set(pessoas_val) - set(pessoas_teste))[:args.pessoas_treino]
    c_treino = ConjuntoDriver(tabela[tabela.pessoa.isin(pessoas_treino)], prep, args.modalidade, treino=True)
    c_val = ConjuntoDriver(tabela[tabela.pessoa.isin(pessoas_val)], prep, args.modalidade)
    c_teste = ConjuntoDriver(tabela[tabela.pessoa.isin(pessoas_teste)], prep, args.modalidade)
    print(f"{nome}: treino {len(c_treino)} ({len(pessoas_treino)} pessoas), val {len(c_val)}, teste {len(c_teste)}", flush=True)

    modelo = M.criar_modelo(args.modelo, 27 if pretreino else len(CLASSES))
    if pretreino:
        estado = torch.load(args.inicio, map_location="cpu", weights_only=False)
        if estado["arquitetura"] != args.modelo:
            raise ValueError("Arquitetura do pré-treino diferente")
        modelo.load_state_dict(estado["modelo"])
        modelo.cabeca = nn.Linear(modelo.cabeca.in_features, len(CLASSES))
    if CANAIS[args.modalidade] != 3:
        modelo = M.adaptar_entrada(modelo, CANAIS[args.modalidade])
    modelo = modelo.cuda()

    lr = args.lr or (3e-4 if pretreino else 1e-3)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-3)
    passos = args.epocas * max(len(c_treino) // args.lote + 1, 1)
    agendador = torch.optim.lr_scheduler.OneCycleLR(otimizador, max_lr=lr, total_steps=passos, pct_start=0.1)
    melhor = -1.0
    historico = pasta / "historico.csv"
    historico.unlink(missing_ok=True)
    with ThreadPoolExecutor(8) as executor:
        for epoca in range(args.epocas):
            modelo.train()
            congelado = pretreino and epoca < args.congelar
            for n, p in modelo.named_parameters():
                p.requires_grad = (not congelado) or n.startswith("cabeca")
            soma, certos, total = 0.0, 0, 0
            for clipes, classes, _ in lotes(c_treino, args.lote, args.seed + epoca, executor, True):
                classes = classes.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    saida = modelo((clipes.cuda(non_blocking=True) - 127.5) / 127.5)
                    perda = nn.functional.cross_entropy(saida.float(), classes, label_smoothing=0.05)
                otimizador.zero_grad(set_to_none=True)
                perda.backward()
                otimizador.step()
                if agendador.last_epoch < passos - 1:
                    agendador.step()
                soma += perda.item() * len(classes); certos += (saida.argmax(1) == classes).sum().item(); total += len(classes)
            if len(c_val):
                p, alvos, _ = prever(modelo, c_val, args.lote, executor)
                acuracia = float((p.argmax(1) == alvos).mean())
            else:
                acuracia = certos / total
            linha = {"epoca": epoca, "perda_treino": soma / total, "acc_treino": certos / total, "acc_val": acuracia}
            with historico.open("a", newline="", encoding="utf-8") as f:
                e = csv.DictWriter(f, fieldnames=list(linha))
                if epoca == 0:
                    e.writeheader()
                e.writerow(linha)
            if epoca % 5 == 0 or epoca == args.epocas - 1:
                print(f" ep{epoca}: treino {certos / total:.3f}, validação {acuracia:.3f}", flush=True)
            if acuracia > melhor:
                melhor = acuracia
                torch.save({"modelo": modelo.state_dict(), "epoca": epoca, "acc_val": melhor, "arquitetura": args.modelo,
                            "classes": CLASSES, "modalidade": args.modalidade}, pasta / "melhor.pt")

        modelo.load_state_dict(torch.load(pasta / "melhor.pt", weights_only=False)["modelo"])
        resultado = {"acc_val": melhor}
        for nome_c, c in (("val", c_val), ("teste", c_teste)):
            if len(c):
                p, alvos, indices = prever(modelo, c, args.lote, executor)
                salvar_predicoes(pasta / f"predicoes_{nome_c}.csv", c, p, alvos, indices)
                resultado[f"acc_{nome_c}"] = float((p.argmax(1) == alvos).mean())
                resultado[f"n_{nome_c}"] = int(len(alvos))
    (pasta / "resultado.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(f"Fim {nome}: {resultado}", flush=True)


if __name__ == "__main__":
    main()
