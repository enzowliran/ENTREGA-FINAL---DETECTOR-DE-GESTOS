"""Baixa do HaGRIDv2 (512 px) só as imagens escolhidas, sem baixar o ZIP inteiro de 128 GB.

O servidor aceita HTTP Range: o script lê o índice do ZIP, sorteia N imagens por classe e por divisão
e busca só os trechos delas. Cada imagem é gravada como data/HaGRID/<divisao>/<classe>/<id>.jpg, com
nome montado aqui, e nunca com o caminho que vem de dentro do ZIP. Quem já foi baixado não é baixado de novo.

Dados: https://github.com/hukenovs/hagrid (licença própria, baseada na CC BY-SA 4.0; citar a fonte).

Uso:
    python src/baixar_hagrid.py --listar
    python src/baixar_hagrid.py --por-classe 500
"""
import argparse
import http.client
import io
import json
import random
import re
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/"
URL_IMAGENS = BASE + "hagridv2_512.zip"
URL_ANOTACOES = BASE + "annotations.zip"  # 99 MB: sem os pontos da mão, que o treino não usa
DIVISOES = ("train", "val", "test")
CLASSES = ("one", "peace", "two_up", "three", "three2", "four", "fist", "like", "dislike", "stop", "palm",
           "no_gesture", "call", "ok", "rock", "mute", "little_finger", "point")
OUTROS_GESTOS = ("call", "ok", "rock", "mute", "little_finger", "point")
ID_VALIDO = re.compile(r"^[0-9a-fA-F-]{8,64}$")


def ler_trecho(url, inicio, fim, tentativas=6):
    """Devolve os bytes de inicio a fim, inclusive, repetindo o pedido se a conexão cair."""
    for tentativa in range(tentativas):
        try:
            pedido = urllib.request.Request(url, headers={"Range": f"bytes={inicio}-{fim}", "User-Agent": "baixar-hagrid"})
            with urllib.request.urlopen(pedido, timeout=60) as resposta:
                if resposta.status != 206:
                    raise OSError(f"o servidor respondeu {resposta.status}, e não 206")
                return resposta.read()
        except (OSError, http.client.HTTPException):
            if tentativa == tentativas - 1:
                raise
            time.sleep(2 ** tentativa)


class ArquivoRemoto(io.RawIOBase):
    """Arquivo somente leitura sobre HTTP, o bastante para o zipfile ler o índice."""

    def __init__(self, url):
        self.url, self.posicao = url, 0
        pedido = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "baixar-hagrid"})
        with urllib.request.urlopen(pedido, timeout=60) as resposta:
            self.tamanho = int(resposta.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.posicao

    def seek(self, deslocamento, origem=0):
        self.posicao = {0: 0, 1: self.posicao, 2: self.tamanho}[origem] + deslocamento
        return self.posicao

    def readinto(self, destino):
        if self.posicao >= self.tamanho or not len(destino):
            return 0
        fim = min(self.posicao + len(destino), self.tamanho) - 1
        dados = ler_trecho(self.url, self.posicao, fim)
        destino[:len(dados)] = dados
        self.posicao += len(dados)
        return len(dados)


def abrir_zip(url):
    # Buffer grande: o índice é lido em sequência, e pedidos pequenos seriam muito lentos.
    return zipfile.ZipFile(io.BufferedReader(ArquivoRemoto(url), buffer_size=16 * 1024 * 1024))


def indice_imagens(pasta):
    """Lista (caminho, início, tamanho comprimido, método) das imagens, com cache em disco."""
    cache = pasta / "indice_imagens.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    t = time.time()
    with abrir_zip(URL_IMAGENS) as z:
        itens = [[i.filename, i.header_offset, i.compress_size, i.compress_type] for i in z.infolist() if not i.is_dir()]
    pasta.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(itens), encoding="utf-8")
    print(f"Índice lido: {len(itens)} arquivos em {time.time() - t:.0f} s", flush=True)
    return itens


def baixar_imagem(item, destino):
    nome, inicio, comprimido, metodo = item
    # O cabeçalho local tem 30 bytes mais nome e extra; a folga evita um segundo pedido.
    dados = ler_trecho(URL_IMAGENS, inicio, inicio + 30 + len(nome.encode()) + 256 + comprimido)
    n_nome, n_extra = struct.unpack("<HH", dados[26:30])
    corpo = dados[30 + n_nome + n_extra:30 + n_nome + n_extra + comprimido]
    if metodo == zipfile.ZIP_DEFLATED:
        corpo = zlib.decompress(corpo, -15)
    elif metodo != zipfile.ZIP_STORED:
        raise ValueError(f"método de compressão inesperado: {metodo}")
    if corpo[:2] != b"\xff\xd8":
        raise ValueError(f"não é JPG: {nome}")
    temporario = destino.with_suffix(".tmp")
    temporario.write_bytes(corpo)
    temporario.replace(destino)
    return len(corpo)


def separar(itens):
    """Agrupa por classe e identificador, aceitando as duas formas de caminho que o ZIP pode ter."""
    por_classe = {}
    for item in itens:
        partes = item[0].replace("\\", "/").split("/")
        arquivo = partes[-1]
        if not arquivo.lower().endswith((".jpg", ".jpeg")) or len(partes) < 2:
            continue
        identificador = arquivo.rsplit(".", 1)[0]
        if ID_VALIDO.match(identificador):
            por_classe.setdefault(partes[-2], []).append((identificador, item))
    return por_classe


def baixar_anotacoes(pasta):
    """Baixa só os JSON das classes usadas, e guarda de cada imagem o que o treino precisa."""
    saida = pasta / "anotacoes.json"
    if saida.exists():
        return json.loads(saida.read_text(encoding="utf-8"))
    anotacoes = {}
    with abrir_zip(URL_ANOTACOES) as z:
        for info in z.infolist():
            partes = info.filename.replace("\\", "/").split("/")
            classe = partes[-1].rsplit(".", 1)[0]
            divisao = next((d for d in DIVISOES if d in partes), None)
            if classe not in CLASSES or divisao is None or not partes[-1].endswith(".json"):
                continue
            t = time.time()
            conteudo = json.loads(z.read(info))
            for identificador, a in conteudo.items():
                anotacoes[identificador] = {"divisao": divisao, "classe": classe, "user_id": a.get("user_id"),
                                            "bboxes": a.get("bboxes"), "labels": a.get("labels")}
            print(f"  {divisao}/{classe}: {len(conteudo)} em {time.time() - t:.0f} s", flush=True)
    saida.write_text(json.dumps(anotacoes), encoding="utf-8")
    return anotacoes


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--destino", type=Path, default=Path("data/HaGRID"))
    a.add_argument("--listar", action="store_true", help="Só mostra como o ZIP está organizado")
    a.add_argument("--por-classe", type=int, default=500, help="Imagens de treino por classe")
    a.add_argument("--fracao-val", type=float, default=0.2, help="Imagens de validação, como fração das de treino")
    a.add_argument("--threads", type=int, default=16)
    a.add_argument("--seed", type=int, default=42)
    args = a.parse_args()

    itens = indice_imagens(args.destino)
    por_classe = separar(itens)
    if args.listar:
        print("Exemplos de caminho:", [i[0] for i in itens[:4]])
        print("Tamanho médio:", sum(i[2] for i in itens) // max(len(itens), 1), "bytes; métodos:", sorted({i[3] for i in itens}))
        for classe, lista in sorted(por_classe.items()):
            print(f"  {classe}: {len(lista)}")
        with abrir_zip(URL_ANOTACOES) as z:
            print("Anotações, exemplos:", [(i.filename, i.file_size) for i in z.infolist()[:6]], "total", len(z.infolist()))
        return

    anotacoes = baixar_anotacoes(args.destino)
    gerador = random.Random(args.seed)
    tarefas = []
    for classe in CLASSES:
        lista = [x for x in por_classe.get(classe, []) if x[0] in anotacoes]
        gerador.shuffle(lista)
        # As 6 classes de "outros" viram uma só no treino, então cada uma precisa de bem menos fotos.
        divisor = len(OUTROS_GESTOS) if classe in OUTROS_GESTOS else 1
        cotas = {"train": args.por_classe // divisor, "val": max(int(args.por_classe * args.fracao_val) // divisor, 60)}
        for divisao, cota in cotas.items():
            escolhidas = [x for x in lista if anotacoes[x[0]]["divisao"] == divisao][:cota]
            pasta = args.destino / divisao / classe
            pasta.mkdir(parents=True, exist_ok=True)
            tarefas += [(item, pasta / f"{identificador}.jpg") for identificador, item in escolhidas
                        if not (pasta / f"{identificador}.jpg").exists()]
    print(f"A baixar: {len(tarefas)} imagens, cerca de {sum(t[0][2] for t in tarefas) / 1e9:.2f} GB", flush=True)

    t, feitos, erros, bytes_total = time.time(), 0, 0, 0
    def tarefa(x):
        return baixar_imagem(*x)
    with ThreadPoolExecutor(args.threads) as executor:
        for futuro in [executor.submit(tarefa, x) for x in tarefas]:
            try:
                bytes_total += futuro.result()
                feitos += 1
            except (OSError, http.client.HTTPException, ValueError, zlib.error) as erro:
                erros += 1
                print("erro:", erro, flush=True)
            if (feitos + erros) % 500 == 0:
                print(f"{feitos + erros}/{len(tarefas)}, {bytes_total / 1e6:.0f} MB, {bytes_total / 1e6 / (time.time() - t):.1f} MB/s, {erros} erros", flush=True)
    print(f"Concluído: {feitos} imagens, {bytes_total / 1e9:.2f} GB, {erros} erros, {time.time() - t:.0f} s", flush=True)
    sys.exit(1 if erros > len(tarefas) * 0.02 else 0)


if __name__ == "__main__":
    main()
