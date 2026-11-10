"""Baixa o IPN Hand (CC BY 4.0) das pastas públicas do Google Drive indicadas na página oficial dos autores.

https://gibranbenitez.github.io/IPN_Hand — Benitez-Garcia et al., "IPN Hand: A Video Dataset and Benchmark for
Real-Time Continuous Hand Gesture Recognition", ICPR 2020.

Lista cada pasta pela visão simples do Drive (embeddedfolderview) e baixa arquivo por arquivo; arquivos já completos
são pulados, então dá para interromper e rodar de novo. Depois extrai os .tgz/.zip na mesma pasta.

Uso:
    python src/baixar_ipn.py                      (anotações, vídeos e máscaras)
    python src/baixar_ipn.py --partes anotacoes   (só as anotações)
"""
import argparse
import html
import re
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

PASTAS = {"anotacoes": "1-mihJEIFoNDpfo1puF8xAMJz6PGVKsBD", "videos": "1O4Fn_jbAEcKIksXHmQIMcHTyaDw4wt9x",
          "mascaras": "1qi5Hm56eUrXCzQVveb1yqFJq6uGQPMDj"}
AGENTE = {"User-Agent": "Mozilla/5.0 (projeto-gestos)"}


def abrir(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=120)


def listar(pasta_id):
    """(nome, id, é_pasta) de cada item da pasta pública."""
    with abrir(f"https://drive.google.com/embeddedfolderview?id={pasta_id}") as r:
        pagina = r.read().decode("utf-8", "replace")
    itens = []
    for bloco in re.findall(r'<div class="flip-entry" id="entry-([\w-]+)".*?<div class="flip-entry-title">(.*?)</div>', pagina, re.S):
        id_, nome = bloco
        e_pasta = f"/drive/folders/{id_}" in pagina
        itens.append((html.unescape(nome).strip(), id_, e_pasta))
    return itens


def baixar(id_, destino):
    url = f"https://drive.usercontent.google.com/download?id={id_}&export=download&confirm=t"
    with abrir(url) as r:
        total = int(r.headers.get("Content-Length") or 0)
        if destino.exists() and total and destino.stat().st_size == total:
            print(f"  já completo: {destino.name}", flush=True)
            return
        if "text/html" in (r.headers.get("Content-Type") or ""):
            raise RuntimeError(f"O Drive devolveu uma página em vez do arquivo {destino.name} (limite de downloads?). Tente mais tarde.")
        temporario = destino.with_suffix(destino.suffix + ".parcial")
        feito, t0, ultimo = 0, time.time(), 0.0
        with temporario.open("wb") as f:
            while bloco := r.read(1 << 20):
                f.write(bloco)
                feito += len(bloco)
                if time.time() - ultimo > 15:
                    ultimo = time.time()
                    vel = feito / max(1e-3, ultimo - t0) / 1e6
                    print(f"  {destino.name}: {feito / 1e9:.2f}" + (f" de {total / 1e9:.2f}" if total else "") + f" GB ({vel:.1f} MB/s)", flush=True)
    temporario.replace(destino)
    print(f"  pronto: {destino.name} ({feito / 1e6:.0f} MB)", flush=True)


def extrair(arquivo, pasta):
    marca = pasta / f".extraido_{arquivo.name}"
    if marca.exists():
        return
    if arquivo.name.endswith((".tgz", ".tar.gz", ".tar")):
        with tarfile.open(arquivo) as t:
            t.extractall(pasta, filter="data")  # filtro "data": recusa caminhos absolutos e links para fora da pasta
    elif arquivo.suffix == ".zip":
        with zipfile.ZipFile(arquivo) as z:
            for nome in z.namelist():
                alvo = (pasta / nome).resolve()
                if not str(alvo).startswith(str(pasta.resolve())):
                    raise RuntimeError(f"Caminho suspeito no zip: {nome}")
            z.extractall(pasta)
    else:
        return
    marca.touch()
    print(f"  extraído: {arquivo.name}", flush=True)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--destino", type=Path, default=Path("data/IPN"))
    a.add_argument("--partes", nargs="+", default=list(PASTAS), choices=list(PASTAS))
    a.add_argument("--nao-extrair", action="store_true")
    args = a.parse_args()
    for parte in args.partes:
        pasta = args.destino / parte
        pasta.mkdir(parents=True, exist_ok=True)
        fila = [(n, i, p, pasta) for n, i, p in listar(PASTAS[parte])]
        if not fila:  # o link é de um arquivo, não de uma pasta (é o caso das máscaras: segment.tgz)
            with abrir(f"https://drive.usercontent.google.com/download?id={PASTAS[parte]}&export=download&confirm=t") as r:
                nome = re.search(r'filename="([^"]+)"', r.headers.get("Content-Disposition", ""))
            fila = [(nome.group(1) if nome else f"{parte}.bin", PASTAS[parte], False, pasta)]
        print(f"{parte}: {len(fila)} itens", flush=True)
        while fila:
            nome, id_, e_pasta, onde = fila.pop(0)
            if e_pasta:
                sub = onde / nome
                sub.mkdir(exist_ok=True)
                fila += [(n, i, p, sub) for n, i, p in listar(id_)]
                continue
            baixar(id_, onde / nome)
            if not args.nao_extrair:
                extrair(onde / nome, onde)


if __name__ == "__main__":
    main()
