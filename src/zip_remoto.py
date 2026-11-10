"""Lê um zip na internet sem baixá-lo inteiro: só o índice e os arquivos pedidos, por pedaços (HTTP Range).

Foi assim que conferimos as listas de classes do Slovo (13 GB) e do ASL Citizen (46 GB) baixando poucos MB.

Uso:
    python src/zip_remoto.py URL --listar                       (mostra os arquivos que não são vídeo ou imagem)
    python src/zip_remoto.py URL --destino data/Slovo --contendo annotations
    python src/zip_remoto.py URL --destino data/Slovo/videos --lista ids.txt   (um nome de arquivo, sem extensão, por linha)
"""
import argparse
import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

MIDIA = (".mp4", ".avi", ".mov", ".webm", ".mkv", ".jpg", ".jpeg", ".png")


class ArquivoRemoto(io.RawIOBase):
    """Arquivo somente leitura em que cada read() vira um pedido HTTP com Range."""

    def __init__(self, url):
        self.url, self.pos, self.baixado = url, 0, 0
        pedido = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "projeto-gestos"})
        with urllib.request.urlopen(pedido, timeout=60) as r:
            self.tamanho = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, pos, de_onde=0):
        self.pos = {0: pos, 1: self.pos + pos, 2: self.tamanho + pos}[de_onde]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.tamanho or len(b) == 0:
            return 0
        fim = min(self.pos + len(b), self.tamanho) - 1
        pedido = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{fim}", "User-Agent": "projeto-gestos"})
        for tentativa in range(5):  # o servidor às vezes derruba a conexão; tenta de novo com espera crescente
            try:
                with urllib.request.urlopen(pedido, timeout=120) as r:
                    dados = r.read()
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if tentativa == 4:
                    raise
                time.sleep(2 * 2 ** tentativa)
        b[:len(dados)] = dados
        self.pos += len(dados)
        self.baixado += len(dados)
        return len(dados)


def abrir(url):
    remoto = ArquivoRemoto(url)
    return remoto, zipfile.ZipFile(io.BufferedReader(remoto, buffer_size=1 << 16))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("url")
    a.add_argument("--listar", action="store_true")
    a.add_argument("--destino", type=Path)
    a.add_argument("--contendo", nargs="*", default=[], help="Extrai os arquivos cujo nome contém algum destes textos")
    a.add_argument("--lista", type=Path, help="Arquivo com os nomes (sem extensão) a extrair, um por linha")
    args = a.parse_args()
    remoto, z = abrir(args.url)
    infos = z.infolist()
    print(f"zip de {remoto.tamanho / 1e9:.2f} GB, {len(infos)} arquivos")
    if args.listar:
        for i in infos:
            if not i.filename.lower().endswith(MIDIA) and not i.is_dir():
                print(f"  {i.filename}  {i.file_size / 1e6:.2f} MB")
    escolhidos = []
    if args.lista:
        nomes = {l.strip() for l in args.lista.read_text(encoding="utf-8").splitlines() if l.strip()}
        escolhidos = [i for i in infos if not i.is_dir() and Path(i.filename).stem in nomes]
        print(f"{len(escolhidos)} de {len(nomes)} nomes encontrados, {sum(i.compress_size for i in escolhidos) / 1e6:.0f} MB")
    if args.contendo:
        escolhidos += [i for i in infos if not i.is_dir() and any(c in i.filename for c in args.contendo)]
    if escolhidos:
        args.destino.mkdir(parents=True, exist_ok=True)
        for k, i in enumerate(escolhidos, 1):
            alvo = args.destino / Path(i.filename).name
            if not alvo.exists():
                alvo.write_bytes(z.read(i))
            if k % 50 == 0:
                print(f"  {k}/{len(escolhidos)}", flush=True)
    print(f"baixado: {remoto.baixado / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
