"""Baixa o Jester e grava os JPG em um único arquivo binário com índice.

O TGZ tem cerca de 5 milhões de JPG pequenos, um volume ruim para extrair no NTFS. Este script lê
as partes como um fluxo contínuo e grava os frames em `jester_frames.bin`, com um índice
`jester_indice.bin`, escrito no disco ao longo do processo. As partes não ficam em disco. A leitura do gzip não permite retomar no meio,
então uma falha definitiva faz o processo recomeçar do início; quedas de conexão são retomadas
com HTTP Range.

Uso:
    python src/baixar_jester.py --destino data/Jester
"""
import argparse
import ctypes
import http.client
import re
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/20bnjester-v1-{:02d}"
TAMANHOS = [10_000_000_000, 10_000_000_000, 2_930_724_987]
BLOCO = 4_000_000
TIPO_INDICE = np.dtype([("video", "<i4"), ("frame", "<i2"), ("deslocamento", "<i8"), ("tamanho", "<i4")])


class FluxoPartes:
    """Expõe as três partes como um único arquivo, retomando a conexão quando ela cai."""

    def __init__(self):
        self.parte, self.posicao, self.resposta, self.total_lido = 0, 0, None, 0

    def _abrir(self):
        pedido = urllib.request.Request(URL.format(self.parte), headers={
            "User-Agent": "python-urllib", "Range": f"bytes={self.posicao}-"})
        self.resposta = urllib.request.urlopen(pedido, timeout=120)
        if self.resposta.status != 206:
            raise RuntimeError(f"O servidor não atendeu o trecho (HTTP {self.resposta.status})")

    def read(self, n=-1):
        n = BLOCO if n is None or n < 0 else n
        falhas = 0
        while self.parte < len(TAMANHOS):
            if self.posicao >= TAMANHOS[self.parte]:
                self.parte, self.posicao, self.resposta = self.parte + 1, 0, None
                continue
            try:
                if self.resposta is None:
                    self._abrir()
                dados = self.resposta.read(n)
                if not dados:
                    raise ConnectionError("conexão encerrada antes do fim da parte")
                self.posicao += len(dados)
                self.total_lido += len(dados)
                return dados
            except (OSError, RuntimeError, http.client.HTTPException) as erro:
                falhas += 1
                self.resposta = None
                if falhas > 60:
                    raise
                espera = min(10 * falhas, 120)
                print(f"  falha de rede ({erro}); nova tentativa em {espera} s", flush=True)
                time.sleep(espera)
        return b""


def main():
    analisador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analisador.add_argument("--destino", type=Path, default=Path("data/Jester"))
    args = analisador.parse_args()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    args.destino.mkdir(parents=True, exist_ok=True)
    final = args.destino / "jester_indice.bin"
    if final.exists():
        print("O índice já existe; nada a fazer.")
        return

    fluxo = FluxoPartes()
    inicio = time.time()
    padrao = re.compile(r"(?:^|/)(\d+)/(\d+)\.jpg$")
    arquivo_bin = args.destino / "jester_frames.bin"
    # Se os frames já existem, esta passagem só reconstrói o índice e confere os bytes.
    conferir = arquivo_bin.exists()
    mapa = np.memmap(arquivo_bin, dtype=np.uint8, mode="r") if conferir else None
    temporario = args.destino / "jester_frames.tmp"
    indice_tmp = args.destino / "jester_indice.tmp"
    n, posicao, videos = 0, 0, set()
    with indice_tmp.open("wb") as indice, tarfile.open(fileobj=fluxo, mode="r|gz") as tar:
        saida = None if conferir else temporario.open("wb")
        for membro in tar:
            achado = padrao.search(membro.name) if membro.isfile() else None
            if not achado:
                if membro.isfile():
                    print("  ignorado:", membro.name, flush=True)
                continue
            dados = tar.extractfile(membro).read()
            if conferir:
                if posicao + len(dados) > mapa.size or (n % 1000 == 0 and bytes(mapa[posicao:posicao + len(dados)]) != dados):
                    raise RuntimeError(f"Frame {membro.name} diferente do arquivo salvo")
            else:
                saida.write(dados)
            registro = np.array([(int(achado[1]), int(achado[2]), posicao, len(dados))], dtype=TIPO_INDICE)
            indice.write(registro.tobytes())
            videos.add(int(achado[1]))
            posicao += len(dados)
            n += 1
            if n % 200_000 == 0:
                indice.flush()
                print(f"{n} frames; baixado {fluxo.total_lido / 1e9:.2f} de 22,93 GB; "
                      f"{(time.time() - inicio) / 60:.0f} min", flush=True)
        if saida is not None:
            saida.close()
    if not conferir:
        temporario.replace(arquivo_bin)
    elif posicao != mapa.size:
        raise RuntimeError(f"Tamanho divergente: {posicao} no arquivo original, {mapa.size} no salvo")
    indice_tmp.replace(final)
    print(f"Fim: {n} frames de {len(videos)} vídeos; {posicao / 1e9:.2f} GB; {(time.time() - inicio) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
