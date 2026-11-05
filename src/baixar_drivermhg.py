"""Baixa o DriverMHG sem a modalidade de profundidade, gerando um RAR por gravação.

O RAR original não é sólido: cada frame é comprimido separadamente. Por isso dá para ler os
cabeçalhos com HTTP Range, pular as sequências de frames de profundidade e copiar os demais
blocos sem alteração. A retomada é por gravação, usando data/DriverMHG/progresso.json.

Frames divididos entre dois volumes do RAR não são copiados: ficam listados no progresso, e o
carregador deve usar o frame vizinho.

Uso:
    python src/baixar_drivermhg.py --destino data/DriverMHG
    python src/baixar_drivermhg.py --destino data/DriverMHG --parar-apos 1
"""
import argparse
import ctypes
import http.client
import json
import re
import struct
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

URL = "https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken%5C{}"
VOLUMES = [("DriverMHG.part1.rar", 5557452800), ("DriverMHG.part2.rar", 5557452800),
           ("DriverMHG.part3.rar", 5557452800), ("DriverMHG.part4.rar", 3277242529)]
PREFIXO_NOME = b"media\\kop\\760E"
BLOCO = 2_000_000
SONDA = 300_000
LIMITE_SEM_BLOCO = 2_000_000
bytes_baixados = 0


def baixar_faixa(volume, inicio, fim, tentativas=60):
    """Baixa os bytes [inicio, fim) de um volume, tentando de novo em falhas de rede."""
    global bytes_baixados
    nome, _ = VOLUMES[volume]
    for tentativa in range(tentativas):
        try:
            pedido = urllib.request.Request(URL.format(nome), headers={
                "User-Agent": "python-urllib", "Range": f"bytes={inicio}-{fim - 1}"})
            with urllib.request.urlopen(pedido, timeout=180) as resposta:
                if resposta.status != 206:
                    raise RuntimeError(f"O servidor não atendeu o trecho (HTTP {resposta.status})")
                dados = resposta.read()
            bytes_baixados += len(dados)
            return dados
        except (OSError, RuntimeError, http.client.HTTPException) as erro:
            if tentativa == tentativas - 1:
                raise
            espera = min(15 * (tentativa + 1), 180)
            print(f"  falha de rede ({erro}); nova tentativa em {espera} s", flush=True)
            time.sleep(espera)


class Leitor:
    """Leitura sequencial de um volume, com um pequeno buffer por trecho."""

    def __init__(self):
        self.volume, self.buffer, self.inicio_buffer = 0, b"", 0

    def ler(self, volume, inicio, n):
        tamanho = VOLUMES[volume][1]
        fim = min(inicio + n, tamanho)
        no_buffer = (self.volume == volume and self.inicio_buffer <= inicio
                     and fim <= self.inicio_buffer + len(self.buffer))
        if not no_buffer:
            self.volume, self.inicio_buffer = volume, inicio
            self.buffer = baixar_faixa(volume, inicio, min(inicio + max(n, BLOCO), tamanho))
        desloc = inicio - self.inicio_buffer
        return self.buffer[desloc:desloc + (fim - inicio)]


def ler_cabecalho(dados):
    """Interpreta um bloco RAR4 no início de `dados`; devolve None se for inválido."""
    if len(dados) < 7:
        return None
    crc, tipo, flags, tamanho = struct.unpack("<HBHH", dados[:7])
    if tamanho < 7 or tamanho > len(dados) or (zlib.crc32(dados[2:tamanho]) & 0xFFFF) != crc:
        return None
    bloco = {"tipo": tipo, "flags": flags, "tamanho": tamanho, "dados": 0, "nome": None}
    if tipo == 0x74:
        empacotado, tam_nome = struct.unpack("<I", dados[7:11])[0], struct.unpack("<H", dados[26:28])[0]
        inicio_nome = 32
        if flags & 0x100:
            empacotado += struct.unpack("<I", dados[32:36])[0] << 32
            inicio_nome = 40
        bloco["dados"] = empacotado
        bloco["nome"] = dados[inicio_nome:inicio_nome + tam_nome].decode("latin1")
    elif flags & 0x8000:
        bloco["dados"] = struct.unpack("<I", dados[7:11])[0]
    return bloco


def identificar(nome):
    """Devolve (sujeito, gravação, modalidade) ou None para arquivos fora das gravações."""
    achado = re.search(r"\\Subject(\d+)\\Rec(\d+)\\(?:([^\\]+)\\)?", nome)
    if not achado:
        return None
    return int(achado[1]), int(achado[2]), (achado[3] or "").lower()


def sondar(volume, posicao):
    """Procura o primeiro cabeçalho de arquivo válido a partir de uma posição do volume."""
    dados = baixar_faixa(volume, posicao, min(posicao + SONDA, VOLUMES[volume][1]))
    i = 0
    while (i := dados.find(PREFIXO_NOME, i)) != -1:
        for inicio_nome in (32, 40):
            if i >= inicio_nome:
                bloco = ler_cabecalho(dados[i - inicio_nome:i - inicio_nome + 512])
                if bloco and bloco["tipo"] == 0x74:
                    return posicao + i - inicio_nome, bloco
        i += 1
    return None


def pular_profundidade(volume, posicao, chave):
    """Encontra um ponto próximo do fim dos frames de profundidade de uma gravação."""
    tamanho = VOLUMES[volume][1]
    ultimo, passo = posicao, 8_000_000
    alto = None
    while alto is None:
        tentativa = ultimo + passo
        if tentativa >= tamanho - SONDA:
            alto = tamanho
            break
        achado = sondar(volume, tentativa)
        if achado and identificar(achado[1]["nome"]) and identificar(achado[1]["nome"])[:2] == chave \
                and identificar(achado[1]["nome"])[2] == "depth":
            ultimo = achado[0]
        else:
            alto = tentativa
    while alto - ultimo > LIMITE_SEM_BLOCO:
        meio = (ultimo + alto) // 2
        achado = sondar(volume, meio)
        if achado and achado[0] < alto and identificar(achado[1]["nome"]) \
                and identificar(achado[1]["nome"])[:2] == chave and identificar(achado[1]["nome"])[2] == "depth":
            ultimo = achado[0]
        else:
            alto = meio
    return ultimo


def fechar_rar(caminho):
    """Acrescenta o bloco de fim de arquivo RAR4."""
    corpo = struct.pack("<BHH", 0x7B, 0x4000, 7)
    with caminho.open("ab") as saida:
        saida.write(struct.pack("<H", zlib.crc32(corpo) & 0xFFFF) + corpo)


def main():
    analisador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analisador.add_argument("--destino", type=Path, default=Path("data/DriverMHG"))
    analisador.add_argument("--parar-apos", type=int, help="Encerra depois de N gravações nesta execução")
    args = analisador.parse_args()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    pasta = args.destino / "gravacoes"
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo_progresso = args.destino / "progresso.json"
    progresso = {"volume": 0, "posicao": 20, "concluidas": [], "frames_divididos": []}
    if arquivo_progresso.exists():
        progresso = json.loads(arquivo_progresso.read_text(encoding="utf-8"))

    leitor = Leitor()
    cabecalho_rar = leitor.ler(0, 0, 20)
    volume, posicao = progresso["volume"], progresso["posicao"]
    atual, saida, parcial = None, None, None
    arquivos_gerais = args.destino / "geral.rar"
    if not arquivos_gerais.exists():
        arquivos_gerais.write_bytes(cabecalho_rar)
    feitas, inicio, saltos_feitos = 0, time.time(), set()

    def concluir(chave, volume_seguinte, posicao_seguinte):
        nonlocal saida, feitas
        saida.close()
        saida = None
        fechar_rar(parcial)
        parcial.replace(parcial.with_suffix(".rar"))
        progresso["concluidas"].append(list(chave))
        progresso["volume"], progresso["posicao"] = volume_seguinte, posicao_seguinte
        arquivo_progresso.write_text(json.dumps(progresso), encoding="utf-8")
        feitas += 1
        print(f"Gravação {chave} pronta ({len(progresso['concluidas'])}/125); baixado nesta execução "
              f"{bytes_baixados / 1e9:.2f} GB em {(time.time() - inicio) / 60:.0f} min", flush=True)

    while volume < len(VOLUMES):
        tamanho_volume = VOLUMES[volume][1]
        if posicao >= tamanho_volume:
            volume, posicao = volume + 1, 20
            continue
        bruto = leitor.ler(volume, posicao, 1024)
        bloco = ler_cabecalho(bruto)
        if bloco is None:
            raise RuntimeError(f"Cabeçalho inválido no volume {volume}, posição {posicao}")
        if bloco["tipo"] == 0x7B:
            volume, posicao = volume + 1, 20
            continue
        total = bloco["tamanho"] + bloco["dados"]
        if bloco["tipo"] != 0x74:
            posicao += total
            continue

        info = identificar(bloco["nome"])
        chave = info[:2] if info else None
        if chave != atual and saida is not None:
            concluir(atual, volume, posicao)
            if args.parar_apos and feitas >= args.parar_apos:
                return
        atual = chave

        if info and list(chave) in progresso["concluidas"]:
            raise RuntimeError(f"Gravação {chave} repetida no arquivo")
        if info and saida is None:
            parcial = pasta / f"Subject{chave[0]:02d}_Rec{chave[1]}.parcial"
            parcial.write_bytes(cabecalho_rar)
            saida = parcial.open("ab")

        dividido = bool(bloco["flags"] & 0x03)
        if info and info[2] == "depth":
            # Salta uma vez por gravação; os frames restantes são percorridos só pelo cabeçalho.
            if chave not in saltos_feitos:
                saltos_feitos.add(chave)
                novo = pular_profundidade(volume, posicao, chave)
                if novo != posicao:
                    posicao = novo
                    continue
            posicao += total
            continue
        if dividido:
            progresso["frames_divididos"].append(bloco["nome"])
            print(f"  frame dividido entre volumes, não copiado: {bloco['nome'][-40:]}", flush=True)
            posicao += total
            continue
        conteudo = leitor.ler(volume, posicao, total)
        if len(conteudo) != total:
            raise RuntimeError(f"Leitura incompleta em {bloco['nome']}")
        if info:
            saida.write(conteudo)
        else:
            with arquivos_gerais.open("ab") as geral:
                geral.write(conteudo)
        posicao += total

    if saida is not None:
        concluir(atual, len(VOLUMES), 0)
    print(f"Fim: {len(progresso['concluidas'])} gravações; baixado {bytes_baixados / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
