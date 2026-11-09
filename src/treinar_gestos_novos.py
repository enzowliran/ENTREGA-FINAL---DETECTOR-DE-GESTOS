"""Acrescenta gestos parados (um a quatro dedos e punho) ao modelo do Jester, sem alterar o modelo de origem.

Os gestos novos vêm do HaGRIDv2, que só tem fotos: cada foto vira um clipe parado, com pequenos tremores.
Para a rede não aprender "esta imagem veio do HaGRID", fotos de gestos que ela já conhece entram nas
classes antigas (joinha, pare), e fotos de outros gestos entram como negativos. O treino mistura essas
fotos com vídeos do Jester, para ela não esquecer o que já sabe. O conjunto de teste não é usado.

Uso:
    python src/treinar_gestos_novos.py --nome cnn_3d_mais_gestos_teste --epocas 3 --jester-por-epoca 6000
"""
import argparse
import csv
import ctypes
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import jester as J
import modelos as M
import treinar_jester as T

NOVAS = ("One Finger", "Two Fingers", "Three Fingers", "Four Fingers", "Fist")
OUTROS = "Doing other things"
MAPA = {"one": "One Finger", "peace": "Two Fingers", "two_up": "Two Fingers", "three": "Three Fingers", "three2": "Three Fingers",
        "four": "Four Fingers", "fist": "Fist", "like": "Thumb Up", "dislike": "Thumb Down", "stop": "Stop Sign", "palm": "Stop Sign",
        "no_gesture": "No gesture", "call": OUTROS, "ok": OUTROS, "rock": OUTROS, "mute": OUTROS, "little_finger": OUTROS, "point": OUTROS}


class ConjuntoHagrid:
    """Cada foto vira um clipe de n_frames: recorte em torno da mão, com tremor de câmera e compressão."""

    def __init__(self, pasta, divisao, anotacoes, nomes, n_frames=16, tamanho=96, treino=True, tremor=0.5, dividir_outros=1):
        self.n_frames, self.tamanho, self.treino, self.tremor = n_frames, tamanho, treino, tremor
        self.amostras = []
        for origem, destino in MAPA.items():
            arquivos = sorted((Path(pasta) / divisao / origem).glob("*.jpg"))
            if destino == OUTROS:
                arquivos = arquivos[::dividir_outros]  # as 6 classes juntas devem pesar como um gesto só
            for arquivo in arquivos:
                a = anotacoes.get(arquivo.stem)
                caixas = [b for b, l in zip(a["bboxes"], a["labels"]) if l == origem] if a else []
                if caixas:
                    self.amostras.append((arquivo, nomes.index(destino), caixas[0], origem))

    def __len__(self):
        return len(self.amostras)

    def carregar(self, posicao, gerador):
        arquivo, classe, (x, y, l, a), _ = self.amostras[posicao]
        imagem = cv2.imdecode(np.fromfile(arquivo, dtype=np.uint8), cv2.IMREAD_COLOR)
        altura, largura = imagem.shape[:2]
        lado_mao = max(l * largura, a * altura)
        cx, cy = (x + l / 2) * largura, (y + a / 2) * altura
        if self.treino:
            # A mão ocupa de 1/4 a 1/2 do quadro, fora do centro, como na webcam e no Jester.
            lado = lado_mao * gerador.uniform(2.0, 4.0)
            cx += lado * gerador.uniform(-0.2, 0.2)
            cy += lado * gerador.uniform(-0.2, 0.2)
            if gerador.random() < 0.5:
                imagem, cx = imagem[:, ::-1], largura - cx  # gesto parado: a mão pode ser qualquer uma
            brilho, qualidade = gerador.uniform(0.8, 1.2), int(gerador.integers(30, 80))
            # Inclui a mão totalmente parada: o tremor não pode virar a pista de que há um gesto.
            tremor = np.cumsum(gerador.normal(0, gerador.choice([0.0, 0.25, 0.5, 1.0]), (self.n_frames, 2)), axis=0)
            ruido = gerador.uniform(0.5, 3.0)
        else:
            lado, brilho, qualidade, ruido = lado_mao * 3.0, 1.0, 60, 1.0
            tremor = np.cumsum(gerador.normal(0, self.tremor, (self.n_frames, 2)), axis=0)
        escala = self.tamanho / max(lado, 1.0)
        quadros = []
        for dx, dy in tremor:
            matriz = np.float32([[escala, 0, self.tamanho / 2 - cx * escala + dx], [0, escala, self.tamanho / 2 - cy * escala + dy]])
            quadros.append(cv2.warpAffine(imagem, matriz, (self.tamanho, self.tamanho), flags=cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE))
        # O Jester é um vídeo pequeno e comprimido; sem isso, a nitidez denunciaria a origem da imagem.
        clipe = np.stack([cv2.imdecode(cv2.imencode(".jpg", q, [cv2.IMWRITE_JPEG_QUALITY, qualidade])[1], cv2.IMREAD_COLOR) for q in quadros])
        clipe = clipe[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) * brilho
        clipe = clipe + gerador.normal(0, ruido, clipe.shape).astype(np.float32)
        return clipe, classe


class ConjuntoMisto:
    """Todas as fotos mais uma parte dos vídeos do Jester, sorteada de novo a cada época."""

    def __init__(self, jester, hagrid, n_jester):
        self.jester, self.hagrid, self.n_jester = jester, hagrid, min(n_jester, len(jester))
        self.sortear(0)

    def sortear(self, epoca):
        self.escolhidos = np.random.default_rng(1000 + epoca).choice(len(self.jester), self.n_jester, replace=False)

    def __len__(self):
        return self.n_jester + len(self.hagrid)

    def carregar(self, posicao, gerador):
        if posicao < self.n_jester:
            return self.jester.carregar(int(self.escolhidos[posicao]), gerador)
        return self.hagrid.carregar(posicao - self.n_jester, gerador)


@torch.no_grad()
def prever(modelo, conjunto, lote):
    modelo.eval()
    probs, classes = [], []
    for clipes, c in J.lotes(conjunto, lote, seed=0, embaralhar=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs.append(modelo(T.preparar(clipes, "cuda")).float().softmax(1).cpu())
        classes.append(c)
    return torch.cat(probs), torch.cat(classes)


def medir(modelo, c_jester, c_hagrid, nomes, lote):
    """Mede o que importa: se o antigo se manteve, se o novo foi aprendido e se o novo dispara sem querer."""
    novas = [nomes.index(n) for n in NOVAS]
    negativos = [nomes.index(n) for n in T.SEM_GESTO]
    gestos = [i for i in range(len(nomes)) if i not in negativos]
    p, c = prever(modelo, c_jester, lote)
    previsto = p.argmax(1)
    neg_j = torch.isin(c, torch.tensor(negativos))
    r = {"jester_top1": float((previsto == c).float().mean()),
         "jester_virou_novo": float(torch.isin(previsto, torch.tensor(novas)).float().mean()),
         "jester_sem_gesto_dispara_07": float((p[neg_j][:, gestos].max(1).values >= 0.7).float().mean())}
    if len(novas) and p.shape[1] > max(novas):
        r["jester_sem_gesto_dispara_novo_07"] = float((p[neg_j][:, novas].max(1).values >= 0.7).float().mean())
    p, c = prever(modelo, c_hagrid, lote)
    previsto = p.argmax(1)
    por_classe = {nomes[i]: float((previsto[c == i] == i).float().mean()) for i in c.unique().tolist()}
    r["hagrid_por_classe"] = por_classe
    r["novas_acerto"] = float(np.mean([por_classe[n] for n in NOVAS if n in por_classe]))
    r["novas_detectadas_07"] = {n: float(((previsto[c == nomes.index(n)] == nomes.index(n)) & (p[c == nomes.index(n)].max(1).values >= 0.7)).float().mean())
                                for n in NOVAS if n in por_classe}
    neg_h = torch.isin(c, torch.tensor(negativos))
    r["hagrid_sem_gesto_dispara_07"] = float((p[neg_h][:, gestos].max(1).values >= 0.7).float().mean())
    matriz = {}
    for n in NOVAS:
        if n in por_classe:
            contagem = np.bincount(previsto[c == nomes.index(n)].numpy(), minlength=len(nomes))
            contagem[nomes.index(n)] = 0
            matriz[n] = nomes[int(contagem.argmax())]
    r["novas_mais_confundidas"] = matriz
    return r


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--origem", default="resultados/jester/cnn_3d/melhor.pt")
    a.add_argument("--nome", required=True)
    a.add_argument("--dados", default="data/Jester")
    a.add_argument("--hagrid", default="data/HaGRID")
    a.add_argument("--epocas", type=int, default=3)
    a.add_argument("--jester-por-epoca", type=int, default=6000)
    a.add_argument("--val-jester", type=int, default=3000, help="Vídeos da validação medidos a cada época")
    a.add_argument("--lote", type=int, default=48)
    a.add_argument("--lr", type=float, default=3e-4, help="Taxa do corpo da rede, já treinado")
    a.add_argument("--lr-cabeca", type=float, default=1e-3, help="Taxa da última camada, que tem saídas novas")
    a.add_argument("--peso-falso", type=float, default=1.0)
    a.add_argument("--suavizacao", type=float, default=0.05)
    # Sem padrão de 0,1 do treino antigo: repetir um frame com a mão ensina que imagem parada é "sem gesto".
    a.add_argument("--frac-sinteticos", type=float, default=0.0)
    a.add_argument("--dividir-outros", type=int, default=6, help="Usa 1 de cada N fotos de outros gestos; use 1 se o download já limitou")
    a.add_argument("--retomar", action="store_true", help="Continua da última época concluída nesta pasta")
    a.add_argument("--seed", type=int, default=42)
    args = a.parse_args()
    pasta = Path("resultados/jester") / args.nome
    if pasta.resolve() == Path(args.origem).resolve().parent:
        raise SystemExit("A pasta de saída não pode ser a do modelo de origem.")
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    estado = torch.load(args.origem, map_location="cuda", weights_only=False)
    nomes_antigos = list(estado["nomes"])
    nomes = nomes_antigos + list(NOVAS)
    base = M.criar_modelo(estado["arquitetura"], len(nomes_antigos)).cuda()
    base.load_state_dict(estado["modelo"])

    _, treino, validacao, _ = J.ler_rotulos(Path(args.dados) / "rotulos/jester_labels")
    negativos = torch.tensor([nomes.index(n) for n in T.SEM_GESTO], device="cuda")
    estaticas = [nomes.index(n) for n in T.ESTATICAS + T.SEM_GESTO]
    c_jester = J.ConjuntoJester(args.dados, treino, treino=True, frac_sinteticos=args.frac_sinteticos, classe_sem_gesto=nomes.index("No gesture"),
                                classes_estaticas=estaticas, aug_tempo=True, aug_espaco=True)
    anotacoes = json.loads((Path(args.hagrid) / "anotacoes.json").read_text(encoding="utf-8"))
    h_treino = ConjuntoHagrid(args.hagrid, "train", anotacoes, nomes, treino=True, dividir_outros=args.dividir_outros)
    h_val = ConjuntoHagrid(args.hagrid, "val", anotacoes, nomes, treino=False, dividir_outros=args.dividir_outros)
    h_parado = ConjuntoHagrid(args.hagrid, "val", anotacoes, nomes, treino=False, tremor=0.0, dividir_outros=args.dividir_outros)
    del anotacoes
    c_treino = ConjuntoMisto(c_jester, h_treino, args.jester_por_epoca)
    # Mesmo sorteio de vídeos em todas as medidas, inclusive na do modelo de origem.
    itens = list(validacao.items())
    random.Random(args.seed).shuffle(itens)
    j_val = J.ConjuntoJester(args.dados, dict(itens[:args.val_jester]), treino=False)
    print(f"Treino: {args.jester_por_epoca} vídeos do Jester por época + {len(h_treino)} fotos. Validação: {len(j_val)} vídeos + {len(h_val)} fotos", flush=True)

    # Referência: o modelo de origem, medido só no Jester, pois ele não tem as classes novas.
    p, c = prever(base, j_val, args.lote)
    referencia = float((p.argmax(1) == c).float().mean())
    neg = torch.isin(c, negativos.cpu())
    gestos_ant = [i for i in range(len(nomes_antigos)) if nomes_antigos[i] not in T.SEM_GESTO]
    ref_falso = float((p[neg][:, gestos_ant].max(1).values >= 0.7).float().mean())
    print(f"Modelo de origem nestes vídeos: top1 {referencia:.4f}; sem gesto que dispara a 0,7: {ref_falso:.4f}", flush=True)

    modelo = M.criar_modelo(estado["arquitetura"], len(nomes)).cuda()
    novo_estado = dict(estado["modelo"])
    cabeca = modelo.state_dict()
    for chave in ("cabeca.weight", "cabeca.bias"):
        # As saídas antigas continuam iguais; as novas partem da inicialização aleatória.
        cabeca[chave][:len(nomes_antigos)] = estado["modelo"][chave]
        novo_estado[chave] = cabeca[chave]
    modelo.load_state_dict(novo_estado)
    del base, estado

    grupos = [{"params": [p for n, p in modelo.named_parameters() if not n.startswith("cabeca")], "lr": args.lr},
              {"params": modelo.cabeca.parameters(), "lr": args.lr_cabeca}]
    otimizador = torch.optim.AdamW(grupos, weight_decay=1e-4)
    passos = args.epocas * (len(c_treino) // args.lote + 1)
    agendador = torch.optim.lr_scheduler.OneCycleLR(otimizador, max_lr=[args.lr, args.lr_cabeca], total_steps=passos, pct_start=0.1)

    historico = pasta / "historico.csv"
    ultimo = pasta / "ultimo.pt"
    melhor, resumos, primeira = -1.0, [], 0
    if args.retomar and ultimo.exists():
        salvo = torch.load(ultimo, map_location="cuda", weights_only=False)
        modelo.load_state_dict(salvo["modelo"]); otimizador.load_state_dict(salvo["otimizador"]); agendador.load_state_dict(salvo["agendador"])
        melhor, primeira = salvo["melhor"], salvo["epoca"] + 1
        resumos = json.loads((pasta / "resumo.json").read_text(encoding="utf-8"))
        print(f"Retomando da época {primeira}", flush=True)
    else:
        historico.unlink(missing_ok=True)
    for epoca in range(primeira, args.epocas):
        c_treino.sortear(epoca)
        modelo.train()
        inicio, soma, certos, total = time.time(), 0.0, 0, 0
        for passo, (clipes, classes) in enumerate(J.lotes(c_treino, args.lote, seed=args.seed + epoca)):
            classes = classes.to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                saida = modelo(T.preparar(clipes, "cuda"))
                perda = nn.functional.cross_entropy(saida.float(), classes, label_smoothing=args.suavizacao)
                perda = perda + args.peso_falso * T.perda_falsa_ativacao(saida.float(), classes, negativos)
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            otimizador.step()
            if agendador.last_epoch < passos - 1:
                agendador.step()
            soma += perda.item() * classes.numel()
            certos += (saida.argmax(1) == classes).sum().item()
            total += classes.numel()
            if passo % 100 == 0:
                print(f" ep{epoca} passo {passo}/{len(c_treino) // args.lote} perda {soma / total:.3f} top1 {certos / total:.3f} "
                      f"{total / (time.time() - inicio):.0f} clipes/s GPU {torch.cuda.max_memory_allocated() >> 20} MB", flush=True)
        duracao = time.time() - inicio
        r = medir(modelo, j_val, h_val, nomes, args.lote)
        p_parado, c_parado = prever(modelo, h_parado, args.lote)
        novas_idx = torch.tensor([nomes.index(n) for n in NOVAS])
        so_novas = torch.isin(c_parado, novas_idx)
        r["novas_acerto_sem_tremor"] = float((p_parado[so_novas].argmax(1) == c_parado[so_novas]).float().mean())
        r.update(epoca=epoca, top1_treino=certos / total, minutos=duracao / 60, referencia_jester_top1=referencia, referencia_sem_gesto_dispara_07=ref_falso)
        resumos.append(r)
        print(f"Época {epoca}: {duracao / 60:.1f} min | Jester top1 {r['jester_top1']:.4f} (origem {referencia:.4f}), virou gesto novo {r['jester_virou_novo']:.4f} | "
              f"novos: acerto {r['novas_acerto']:.4f}, sem tremor {r['novas_acerto_sem_tremor']:.4f} | sem gesto dispara a 0,7: Jester {r['jester_sem_gesto_dispara_07']:.4f} (origem {ref_falso:.4f}), "
              f"gestos novos {r['jester_sem_gesto_dispara_novo_07']:.4f}, HaGRID {r['hagrid_sem_gesto_dispara_07']:.4f}", flush=True)
        print("   por classe:", {k: round(v, 3) for k, v in r["hagrid_por_classe"].items()}, flush=True)
        print("   detectados a 0,7:", {k: round(v, 3) for k, v in r["novas_detectadas_07"].items()}, "| confunde:", r["novas_mais_confundidas"], flush=True)
        linha = {k: v for k, v in r.items() if isinstance(v, (int, float))}
        novo = not historico.exists()
        with historico.open("a", newline="", encoding="utf-8") as arquivo:
            e = csv.DictWriter(arquivo, fieldnames=list(linha))
            if novo:
                e.writeheader()
            e.writerow(linha)
        (pasta / "resumo.json").write_text(json.dumps(resumos, indent=2, ensure_ascii=False), encoding="utf-8")
        # Escolha só pela validação: o antigo e o novo valem o mesmo.
        nota = (r["jester_top1"] + r["novas_acerto"]) / 2
        if nota > melhor:
            melhor = nota
            torch.save({"modelo": modelo.state_dict(), "nomes": nomes, "epoca": epoca, "top1_val": r["jester_top1"],
                        "arquitetura": "cnn_3d", "origem": args.origem}, pasta / "melhor.pt")
        torch.save({"modelo": modelo.state_dict(), "otimizador": otimizador.state_dict(), "agendador": agendador.state_dict(),
                    "epoca": epoca, "melhor": melhor}, ultimo)


if __name__ == "__main__":
    main()
