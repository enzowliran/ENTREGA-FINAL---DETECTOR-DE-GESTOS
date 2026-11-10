"""Retreina um e dois dedos com vídeo real, partindo do modelo de 32 saídas (cnn_3d_mais_gestos).

O que mudou em relação ao Passo 6 (treinar_gestos_novos.py), pelo diagnóstico de 10/11:
  - entram vídeos reais (data/videos_reais, de src/preparar_videos_reais.py): IPN Hand com a mão quase parada,
    "sem gesto" do IPN e do Slovo, e cliques do IPN como "outro movimento";
  - as fotos do HaGRID ganham movimento natural (trajetória suave, leve zoom e rotação, desfoque), em vez do
    tremor quadro a quadro, que deixava os dedos com "cara de foto";
  - a escolha do modelo é feita em vídeo real de pessoas fora do treino (IPN, 13 pessoas; sinais "um" e "dois" do
    Slovo e do ASL Citizen), nunca em foto. Os gestos antigos não podem cair mais que 1 ponto no Jester.

Uso:
    python src/treinar_dedos.py --nome cnn_3d_dedos
    python src/treinar_dedos.py --nome cnn_3d_dedos --retomar
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
from treinar_gestos_novos import ConjuntoHagrid, prever

DEDOS = ("One Finger", "Two Fingers")


class HagridNatural(ConjuntoHagrid):
    """Fotos do HaGRID com movimento de câmera e de mão mais parecido com vídeo real."""

    def carregar(self, posicao, gerador):
        if not self.treino:
            return super().carregar(posicao, gerador)
        arquivo, classe, (x, y, l, a), _ = self.amostras[posicao]
        imagem = cv2.imdecode(np.fromfile(arquivo, dtype=np.uint8), cv2.IMREAD_COLOR)
        altura, largura = imagem.shape[:2]
        lado_mao = max(l * largura, a * altura)
        cx, cy = (x + l / 2) * largura, (y + a / 2) * altura
        if gerador.random() < 0.5:
            imagem, cx = imagem[:, ::-1], largura - cx
        lado = lado_mao * gerador.uniform(2.0, 4.5)
        cx += lado * gerador.uniform(-0.2, 0.2)
        cy += lado * gerador.uniform(-0.2, 0.2)
        n, s = self.n_frames, np.linspace(0, 1, self.n_frames)
        suave = s * s * (3 - 2 * s)
        parado = gerador.random() < 0.15  # alguns clipes bem parados, como uma mão firme
        amp = 0.0 if parado else gerador.uniform(0.0, 0.10)
        d0, d1 = gerador.uniform(-amp, amp, 2) * lado, gerador.uniform(-amp, amp, 2) * lado
        z0, z1 = (1.0, 1.0) if parado else gerador.uniform(0.94, 1.06, 2)
        r0, r1 = (0.0, 0.0) if parado else gerador.uniform(-6, 6, 2)
        onda, freq, fase = (0.0 if parado else gerador.uniform(0, 0.012) * lado), gerador.uniform(0.5, 2.0), gerador.uniform(0, 6.3)
        quadros, anterior = [], None
        for k in range(n):
            d = d0 + (d1 - d0) * suave[k] + onda * np.array([np.sin(2 * np.pi * freq * k / n + fase), np.cos(2 * np.pi * freq * k / n + fase)])
            esc = self.tamanho / lado * (z0 + (z1 - z0) * suave[k])
            ang = np.deg2rad(r0 + (r1 - r0) * suave[k])
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * esc
            centro = np.array([cx, cy]) + d
            t = self.tamanho / 2 - R @ centro
            matriz = np.float32([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]]])
            q = cv2.warpAffine(imagem, matriz, (self.tamanho, self.tamanho), flags=cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            if anterior is not None:  # desfoque na direção do movimento, quando a mão anda mais de 1,5 px no quadro
                passo = (centro - anterior) * esc
                comprimento = int(np.hypot(*passo))
                if comprimento >= 2:
                    nucleo = np.zeros((comprimento * 2 + 1, comprimento * 2 + 1), np.float32)
                    cv2.line(nucleo, (comprimento - int(passo[0] / 2), comprimento - int(passo[1] / 2)),
                             (comprimento + int(passo[0] / 2), comprimento + int(passo[1] / 2)), 1.0, 1)
                    q = cv2.filter2D(q, -1, nucleo / max(nucleo.sum(), 1))
            anterior = centro
            quadros.append(q)
        qualidade, brilho, ruido = int(gerador.integers(30, 80)), gerador.uniform(0.8, 1.2), gerador.uniform(0.5, 3.0)
        clipe = np.stack([cv2.imdecode(cv2.imencode(".jpg", q, [cv2.IMWRITE_JPEG_QUALITY, qualidade])[1], cv2.IMREAD_COLOR) for q in quadros])
        clipe = clipe[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) * brilho
        return clipe + gerador.normal(0, ruido, clipe.shape).astype(np.float32), classe


class ConjuntoReais:
    """Janelas de 1 s (ou mais, nos negativos) dos vídeos reais preparados."""

    def __init__(self, pasta, trechos, nomes, treino, por_classe=None, n_frames=16, tamanho=96, janelas_validacao=4):
        self.mapa = np.memmap(Path(pasta) / "quadros.bin", dtype=np.uint8, mode="r")
        self.trechos = [t for t in trechos if t["janelas"]]
        self.nomes, self.treino, self.n_frames, self.tamanho = nomes, treino, n_frames, tamanho
        self.por_classe = por_classe or {}
        self.candidatos = {}
        for i, t in enumerate(self.trechos):
            self.candidatos.setdefault(t["classe"], []).extend((i, k) for k in t["janelas"])
        if treino:
            self.sortear(0)
        else:  # validação fixa: até N janelas espalhadas por trecho; nos sinais, o trecho inteiro
            self.amostras = []
            for i, t in enumerate(self.trechos):
                if t["divisao"] == "validacao_sinais":
                    self.amostras.append((i, -1))
                else:
                    js = t["janelas"]
                    self.amostras += [(i, js[int(p)]) for p in np.linspace(0, len(js) - 1, min(janelas_validacao, len(js)))]

    def sortear(self, epoca):
        g = np.random.default_rng(5000 + epoca)
        self.amostras = []
        for classe, n in self.por_classe.items():
            c = self.candidatos.get(classe, [])
            if c:
                self.amostras += [c[j] for j in g.integers(0, len(c), n)]

    def __len__(self):
        return len(self.amostras)

    def classe(self, posicao):
        return self.nomes.index(self.trechos[self.amostras[posicao][0]]["classe"])

    def carregar(self, posicao, gerador):
        i, k = self.amostras[posicao]
        t = self.trechos[i]
        total = len(t["quadros"])
        if k < 0:
            inicio, comprimento = 0, total
        elif t["classe"] in DEDOS:
            inicio, comprimento = k, min(12, total - k)  # a janela de 1 s que passou no critério de mão quase parada
        else:
            inicio = k
            comprimento = int(gerador.integers(12, 37)) if self.treino else 18
            comprimento = min(comprimento, total - k)
        pontos = inicio + np.linspace(0, comprimento, self.n_frames + 1)
        if self.treino:
            indices = (pontos[:-1] + gerador.random(self.n_frames) * (pontos[1:] - pontos[:-1])).astype(int)
        else:
            indices = ((pontos[:-1] + pontos[1:]) / 2).astype(int)
        indices = np.minimum(indices, total - 1)
        imagens = [cv2.imdecode(np.asarray(self.mapa[t["quadros"][j][0]:t["quadros"][j][0] + t["quadros"][j][1]]), cv2.IMREAD_COLOR) for j in indices]
        lado = imagens[0].shape[0]
        if self.treino:
            recorte = int(lado * gerador.uniform(0.8, 1.0))
            y0, x0 = int(gerador.integers(0, lado - recorte + 1)), int(gerador.integers(0, lado - recorte + 1))
            brilho, espelho = gerador.uniform(0.8, 1.2), gerador.random() < 0.5
        else:
            recorte, y0, x0, brilho, espelho = lado, 0, 0, 1.0, False
        clipe = np.stack([cv2.resize(im[y0:y0 + recorte, x0:x0 + recorte], (self.tamanho, self.tamanho), interpolation=cv2.INTER_AREA) for im in imagens])
        if espelho:  # um ou dois dedos parados, sem gesto e cliques não mudam de classe ao espelhar
            clipe = clipe[:, :, ::-1]
        return clipe[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) * brilho, self.nomes.index(t["classe"])


class Mistura:
    """Várias fontes, cada uma com quantas amostras entram por época (sorteadas de novo a cada época)."""

    def __init__(self, partes):
        self.partes = partes  # [(conjunto, n por época ou None para todas)]
        self.sortear(0)

    def sortear(self, epoca):
        g = np.random.default_rng(1000 + epoca)
        self.indice = []
        for p, (conjunto, n) in enumerate(self.partes):
            if hasattr(conjunto, "sortear") and isinstance(conjunto, ConjuntoReais):
                conjunto.sortear(epoca)
            total = len(conjunto)
            escolhidos = np.arange(total) if n is None or n >= total else g.choice(total, n, replace=False)
            self.indice += [(p, int(e)) for e in escolhidos]

    def __len__(self):
        return len(self.indice)

    def carregar(self, posicao, gerador):
        p, e = self.indice[posicao]
        return self.partes[p][0].carregar(e, gerador)


def medir_reais(modelo, conjunto, nomes, lote, prefixo):
    """Acerto em um e dois dedos, e quanto um e dois dedos disparam (>= 0,7) nos vídeos sem gesto."""
    p, c = prever(modelo, conjunto, lote)
    dedos = [nomes.index(d) for d in DEDOS]
    r = {}
    for d in DEDOS:
        sel = c == nomes.index(d)
        if sel.any():
            pd = p[sel]
            posicao = (pd > pd[:, [nomes.index(d)]]).sum(1) + 1
            r[f"{prefixo}_{d}_top1"] = float((posicao == 1).float().mean())
            r[f"{prefixo}_{d}_top5"] = float((posicao <= 5).float().mean())
            r[f"{prefixo}_{d}_dispara07"] = float(((pd.argmax(1) == nomes.index(d)) & (pd.max(1).values >= 0.7)).float().mean())
            r[f"{prefixo}_{d}_n"] = int(sel.sum())
    neg = torch.isin(c, torch.tensor([nomes.index(n) for n in T.SEM_GESTO]))
    if neg.any():
        pn = p[neg]
        r[f"{prefixo}_sem_gesto_vira_dedo07"] = float((torch.isin(pn.argmax(1), torch.tensor(dedos)) & (pn.max(1).values >= 0.7)).float().mean())
        r[f"{prefixo}_sem_gesto_n"] = int(neg.sum())
    return r


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--origem", default="resultados/jester/cnn_3d_mais_gestos/melhor.pt")
    a.add_argument("--nome", required=True)
    a.add_argument("--dados", default="data/Jester")
    a.add_argument("--hagrid", default="data/HaGRID")
    a.add_argument("--reais", default="data/videos_reais")
    a.add_argument("--epocas", type=int, default=8)
    a.add_argument("--jester-por-epoca", type=int, default=60000)
    a.add_argument("--hagrid-por-epoca", type=int, default=25000, help="Fotos por época (de 49 mil): o vídeo real deve pesar mais")
    a.add_argument("--reais-dedo", type=int, default=8000, help="Janelas de vídeo real por época, para cada dedo")
    a.add_argument("--reais-sem-gesto", type=int, default=5000)
    a.add_argument("--reais-outros", type=int, default=3000)
    a.add_argument("--val-jester", type=int, default=3000)
    a.add_argument("--lote", type=int, default=48)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--lr-cabeca", type=float, default=1e-3)
    a.add_argument("--peso-falso", type=float, default=1.0)
    a.add_argument("--suavizacao", type=float, default=0.05)
    a.add_argument("--queda-max-jester", type=float, default=0.01, help="Queda máxima aceita no Jester em relação à origem")
    a.add_argument("--retomar", action="store_true")
    a.add_argument("--teste-rapido", action="store_true", help="Poucos passos e validação pequena, só para conferir que tudo roda")
    a.add_argument("--seed", type=int, default=42)
    args = a.parse_args()
    pasta = Path("resultados/jester") / args.nome
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # não deixa o Windows dormir no meio
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    estado = torch.load(args.origem, map_location="cuda", weights_only=False)
    nomes = list(estado["nomes"])
    modelo = M.criar_modelo(estado["arquitetura"], len(nomes)).cuda()
    modelo.load_state_dict(estado["modelo"])
    negativos = torch.tensor([nomes.index(n) for n in T.SEM_GESTO], device="cuda")

    _, treino, validacao, _ = J.ler_rotulos(Path(args.dados) / "rotulos/jester_labels")
    estaticas = [nomes.index(n) for n in T.ESTATICAS + T.SEM_GESTO]
    c_jester = J.ConjuntoJester(args.dados, treino, treino=True, frac_sinteticos=0.0, classe_sem_gesto=nomes.index("No gesture"),
                                classes_estaticas=estaticas, aug_tempo=True, aug_espaco=True)
    anotacoes = json.loads((Path(args.hagrid) / "anotacoes.json").read_text(encoding="utf-8"))
    h_treino = HagridNatural(args.hagrid, "train", anotacoes, nomes, treino=True, dividir_outros=1)
    del anotacoes
    trechos = json.loads((Path(args.reais) / "trechos.json").read_text(encoding="utf-8"))
    r_treino = ConjuntoReais(args.reais, [t for t in trechos if t["divisao"] == "treino"], nomes, treino=True,
                             por_classe={"One Finger": args.reais_dedo, "Two Fingers": args.reais_dedo,
                                         "No gesture": args.reais_sem_gesto, "Doing other things": args.reais_outros})
    v_ipn = ConjuntoReais(args.reais, [t for t in trechos if t["divisao"] == "validacao" and t["base"] == "IPN"], nomes, treino=False)
    v_slovo = ConjuntoReais(args.reais, [t for t in trechos if t["divisao"] == "validacao" and t["base"] == "Slovo"], nomes, treino=False, janelas_validacao=2)
    v_sinais = ConjuntoReais(args.reais, [t for t in trechos if t["divisao"] == "validacao_sinais"], nomes, treino=False)
    itens = list(validacao.items())
    random.Random(args.seed).shuffle(itens)
    j_val = J.ConjuntoJester(args.dados, dict(itens[:args.val_jester if not args.teste_rapido else 300]), treino=False)
    if args.teste_rapido:
        v_ipn.amostras, v_slovo.amostras = v_ipn.amostras[::20], v_slovo.amostras[::10]
    c_treino = Mistura([(c_jester, args.jester_por_epoca), (h_treino, args.hagrid_por_epoca), (r_treino, None)])
    print(f"Treino por época: {args.jester_por_epoca} Jester + {args.hagrid_por_epoca} fotos + {len(r_treino)} janelas reais = {len(c_treino)}", flush=True)
    print(f"Validação: {len(j_val)} Jester, {len(v_ipn)} IPN (13 pessoas), {len(v_slovo)} Slovo sem gesto, {len(v_sinais)} sinais um/dois", flush=True)

    def validar():
        p, c = prever(modelo, j_val, args.lote)
        r = {"jester_top1": float((p.argmax(1) == c).float().mean())}
        for conj, prefixo in ((v_ipn, "ipn"), (v_slovo, "slovo"), (v_sinais, "sinais")):
            r.update(medir_reais(modelo, conj, nomes, args.lote, prefixo))
        # nota dos dedos em vídeo real: média do 1º lugar no IPN (pessoas fora do treino) e nos sinais
        partes = [r.get(f"{pre}_{d}_top1") for pre in ("ipn", "sinais") for d in DEDOS]
        r["dedos_real"] = float(np.mean([x for x in partes if x is not None]))
        return r

    origem = validar()
    print("Modelo de origem: " + " | ".join(f"{k} {v:.3f}" for k, v in origem.items() if not k.endswith("_n")), flush=True)
    (pasta / "origem.json").write_text(json.dumps(origem, indent=2), encoding="utf-8")

    novo = dict(estado["modelo"])
    modelo.load_state_dict(novo)
    grupos = [{"params": [p for n, p in modelo.named_parameters() if not n.startswith("cabeca")], "lr": args.lr},
              {"params": modelo.cabeca.parameters(), "lr": args.lr_cabeca}]
    otimizador = torch.optim.AdamW(grupos, weight_decay=1e-4)
    passos_epoca = len(c_treino) // args.lote + 1 if not args.teste_rapido else 30
    passos = args.epocas * passos_epoca
    agendador = torch.optim.lr_scheduler.OneCycleLR(otimizador, max_lr=[args.lr, args.lr_cabeca], total_steps=passos, pct_start=0.1)

    historico, ultimo = pasta / "historico.csv", pasta / "ultimo.pt"
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
            if passo >= passos_epoca:
                break
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
            if passo % 200 == 0:
                print(f" ep{epoca} passo {passo}/{passos_epoca} perda {soma / total:.3f} top1 {certos / total:.3f} {total / (time.time() - inicio):.0f} clipes/s", flush=True)
        duracao = time.time() - inicio
        r = validar()
        r.update(epoca=epoca, top1_treino=certos / total, minutos=duracao / 60)
        resumos.append(r)
        pct = lambda k: f"{100 * r[k]:.1f}%" if k in r else "—"
        print(f"Época {epoca + 1}/{args.epocas} ({duracao / 60:.1f} min) | Jester {pct('jester_top1')} (origem {100 * origem['jester_top1']:.1f}%) | "
              f"IPN um {pct('ipn_One Finger_top1')} dois {pct('ipn_Two Fingers_top1')} | sinais um {pct('sinais_One Finger_top1')} dois {pct('sinais_Two Fingers_top1')} | "
              f"top5 sinais um {pct('sinais_One Finger_top5')} dois {pct('sinais_Two Fingers_top5')} | sem gesto vira dedo: IPN {pct('ipn_sem_gesto_vira_dedo07')} Slovo {pct('slovo_sem_gesto_vira_dedo07')}", flush=True)
        linha = {k: v for k, v in r.items() if isinstance(v, (int, float))}
        with historico.open("a", newline="", encoding="utf-8") as f:
            e = csv.DictWriter(f, fieldnames=list(linha))
            if f.tell() == 0:
                e.writeheader()
            e.writerow(linha)
        (pasta / "resumo.json").write_text(json.dumps(resumos, indent=2, ensure_ascii=False), encoding="utf-8")
        # Escolha: o melhor em dedos no vídeo real, desde que os gestos antigos não caiam mais que o limite.
        if r["jester_top1"] >= origem["jester_top1"] - args.queda_max_jester and r["dedos_real"] > melhor:
            melhor = r["dedos_real"]
            torch.save({"modelo": modelo.state_dict(), "nomes": nomes, "epoca": epoca, "top1_val": r["jester_top1"], "arquitetura": estado["arquitetura"],
                        "origem": args.origem}, pasta / "melhor.pt")
            print(f"   salvo como melhor (dedos em vídeo real {100 * melhor:.1f}%)", flush=True)
        torch.save({"modelo": modelo.state_dict(), "otimizador": otimizador.state_dict(), "agendador": agendador.state_dict(),
                    "epoca": epoca, "melhor": melhor}, ultimo)


if __name__ == "__main__":
    main()
