"""Carregador do Jester a partir de jester_frames.bin, decodificando os JPG em threads.

O cv2 libera o GIL ao decodificar, então threads bastam e evitam o custo de abrir processos no
Windows. Os lotes são montados em segundo plano e entregues já como tensores.
"""
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

TIPO_INDICE = np.dtype([("video", "<i4"), ("frame", "<i2"), ("deslocamento", "<i8"), ("tamanho", "<i4")])


def ler_rotulos(pasta):
    """Devolve os nomes das classes e as tabelas {vídeo: classe} de cada divisão."""
    pasta = Path(pasta)
    nomes = [l.strip() for l in (pasta / "labels.csv").read_text().splitlines() if l.strip()]
    indice = {n: i for i, n in enumerate(nomes)}

    def ler(arquivo):
        tabela = {}
        for linha in (pasta / arquivo).read_text().splitlines():
            if ";" in linha:
                video, nome = linha.split(";", 1)
                tabela[int(video)] = indice[nome.strip()]
        return tabela

    return nomes, ler("train.csv"), ler("validation.csv"), ler("test-answers.csv")


class ConjuntoJester:
    def __init__(self, pasta, rotulos, n_frames=16, tamanho=96, treino=True, frac_sinteticos=0.0,
                 classe_sem_gesto=None, classes_estaticas=(), aug_tempo=False, aug_espaco=False):
        # Sorteia um trecho e uma velocidade, para a rede reconhecer gestos curtos na janela.
        self.aug_tempo = aug_tempo
        # Reduz e desloca a cena, para a mão aparecer menor e fora do centro.
        self.aug_espaco = aug_espaco
        # Fração das amostras de treino trocadas por negativos sintéticos (sem movimento ou ruído).
        self.frac_sinteticos, self.classe_sem_gesto = frac_sinteticos, classe_sem_gesto
        self.classes_estaticas = set(classes_estaticas)
        pasta = Path(pasta)
        dados = np.fromfile(pasta / "jester_indice.bin", dtype=TIPO_INDICE)
        ordem = np.lexsort((dados["frame"], dados["video"]))
        video = dados["video"][ordem]
        self.deslocamento, self.tamanho_jpg = dados["deslocamento"][ordem], dados["tamanho"][ordem]
        ids, inicios, contagens = np.unique(video, return_index=True, return_counts=True)
        faixas = {int(v): (int(i), int(c)) for v, i, c in zip(ids, inicios, contagens)}
        self.amostras = [(v, *faixas[v], c) for v, c in rotulos.items() if v in faixas]
        self.faltando = len(rotulos) - len(self.amostras)
        self.n_frames, self.tamanho, self.treino = n_frames, tamanho, treino
        self.mapa = np.memmap(pasta / "jester_frames.bin", dtype=np.uint8, mode="r")

    def __len__(self):
        return len(self.amostras)

    def carregar(self, posicao, gerador):
        _, inicio, contagem, classe = self.amostras[posicao]
        base, extensao = 0.0, float(contagem)
        if self.treino and self.aug_tempo:
            extensao = contagem * gerador.uniform(0.4, 1.0)
            base = gerador.uniform(0, contagem - extensao)
        pontos = base + np.linspace(0, extensao, self.n_frames + 1)
        if self.treino:
            indices = (pontos[:-1] + gerador.random(self.n_frames) * (pontos[1:] - pontos[:-1])).astype(int)
        else:
            indices = ((pontos[:-1] + pontos[1:]) / 2).astype(int)
        indices = np.minimum(indices, contagem - 1) + inicio

        imagens = []
        for i in indices:
            d, t = self.deslocamento[i], self.tamanho_jpg[i]
            imagens.append(cv2.imdecode(np.asarray(self.mapa[d:d + t]), cv2.IMREAD_COLOR))
        altura, largura = imagens[0].shape[:2]
        lado = min(altura, largura)
        if self.treino:
            # Sem inversão horizontal: ela trocaria a classe de gestos como deslizar para a esquerda.
            recorte = int(lado * gerador.uniform(0.8, 1.0))
            y0 = gerador.integers(0, altura - recorte + 1)
            x0 = gerador.integers(0, largura - recorte + 1)
            brilho = gerador.uniform(0.8, 1.2)
        else:
            recorte, y0, x0, brilho = lado, (altura - lado) // 2, (largura - lado) // 2, 1.0
        clipe = np.stack([cv2.resize(im[y0:y0 + recorte, x0:x0 + recorte], (self.tamanho, self.tamanho),
                                     interpolation=cv2.INTER_LINEAR) for im in imagens])
        if self.treino and self.aug_espaco:
            escala = gerador.uniform(0.5, 1.0)
            lado_novo = max(int(self.tamanho * escala), 8)
            folga = self.tamanho - lado_novo
            y, x = int(gerador.integers(0, folga + 1)), int(gerador.integers(0, folga + 1))
            matriz = np.float32([[escala, 0, x], [0, escala, y]])
            clipe = np.stack([cv2.warpAffine(q, matriz, (self.tamanho, self.tamanho), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REPLICATE) for q in clipe])
        clipe = clipe[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) * brilho
        if self.treino and self.frac_sinteticos and classe not in self.classes_estaticas                 and gerador.random() < self.frac_sinteticos:
            sorteio = gerador.random()
            if sorteio < 0.7:
                # Um único frame repetido: a mão pode aparecer, mas não há movimento.
                clipe = np.repeat(clipe[gerador.integers(0, self.n_frames)][None], self.n_frames, axis=0)
                clipe = clipe + gerador.normal(0, 2.0, clipe.shape).astype(np.float32)
            elif sorteio < 0.85:
                clipe = gerador.uniform(0, 255, clipe.shape).astype(np.float32)
            else:
                clipe = np.full(clipe.shape, gerador.uniform(0, 255), dtype=np.float32)
            classe = self.classe_sem_gesto
        return clipe, classe


def lotes(conjunto, tamanho_lote, seed, n_threads=14, prefetch=6, embaralhar=True):
    """Gera (clipes, classes) em tensores de CPU, montados em segundo plano."""
    ordem = np.random.default_rng(seed).permutation(len(conjunto)) if embaralhar else np.arange(len(conjunto))
    fila = queue.Queue(maxsize=prefetch)
    FIM = object()

    def tarefa(args):
        posicao, semente = args
        return conjunto.carregar(posicao, np.random.default_rng(semente))

    def produtor():
        try:
            with ThreadPoolExecutor(n_threads) as executor:
                for inicio in range(0, len(ordem), tamanho_lote):
                    parte = ordem[inicio:inicio + tamanho_lote]
                    resultados = list(executor.map(tarefa, [(int(p), seed * 1_000_003 + int(p)) for p in parte]))
                    clipes = np.stack([r[0] for r in resultados])
                    classes = np.array([r[1] for r in resultados])
                    fila.put((torch.from_numpy(clipes), torch.from_numpy(classes)))
        except Exception as erro:
            fila.put(erro)
        fila.put(FIM)

    threading.Thread(target=produtor, daemon=True).start()
    while (item := fila.get()) is not FIM:
        if isinstance(item, Exception):
            raise item
        yield item
