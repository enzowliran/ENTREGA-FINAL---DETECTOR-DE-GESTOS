"""Converte cada gravação do DriverMHG em matrizes uint8 de acesso rápido.

Gera `prep/S01_R1_rgb.npy` (T, 96, 128, 3) e `prep/S01_R1_ir.npy` (T, 96, 128). A proporção 4:3 é
mantida e as anotações são copiadas. Os JPG temporários são apagados ao final de cada gravação.
Frames ausentes, como os divididos entre volumes do RAR, repetem o frame anterior.

Uso:
    python src/preparar_drivermhg.py --base data/DriverMHG
"""
import argparse
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

LARGURA, ALTURA = 128, 96


def _ler(args):
    caminho, modo = args
    imagem = cv2.imread(str(caminho), modo)
    if imagem is None:
        return None
    return cv2.resize(imagem, (LARGURA, ALTURA), interpolation=cv2.INTER_AREA)


def converter_modalidade(pasta, modo, executor):
    arquivos = {int(p.stem): p for p in pasta.glob("*.jpg")}
    n = max(arquivos) + 1
    imagens = list(executor.map(_ler, [(arquivos.get(i, Path("ausente")), modo) for i in range(n)]))
    ausentes = sum(i is None for i in imagens)
    for i in range(n):
        if imagens[i] is None:
            imagens[i] = imagens[i - 1] if i else next(x for x in imagens if x is not None)
    return np.stack(imagens), ausentes


def converter(rar, base, executor):
    achado = re.match(r"Subject(\d+)_Rec(\d+)", rar.stem)
    nome = f"S{int(achado[1]):02d}_R{achado[2]}"
    destino = base / "prep"
    destino.mkdir(exist_ok=True)
    if (destino / f"{nome}_ir.npy").exists():
        return False
    temporario = base / f"tmp_{nome}"
    temporario.mkdir(exist_ok=True)
    subprocess.run([r"C:\Windows\System32\tar.exe", "-xf", str(rar), "-C", str(temporario)], check=False, capture_output=True)
    gravacao = next(p for p in temporario.rglob(f"Rec{achado[2]}") if p.is_dir())
    rgb, faltas_rgb = converter_modalidade(gravacao / "rgb", cv2.IMREAD_COLOR, executor)
    ir, faltas_ir = converter_modalidade(gravacao / "ir", cv2.IMREAD_GRAYSCALE, executor)
    np.save(destino / f"{nome}_rgb.npy", rgb[..., ::-1])
    shutil.copy(gravacao / "annotationall_all.txt", destino / f"{nome}_anotacao.txt")
    np.save(destino / f"{nome}_ir.npy", ir)
    shutil.rmtree(temporario)
    print(f"{nome}: {len(rgb)} frames RGB, {len(ir)} IR, ausentes {faltas_rgb}/{faltas_ir}", flush=True)
    return True


def main():
    analisador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analisador.add_argument("--base", type=Path, default=Path("data/DriverMHG"))
    args = analisador.parse_args()
    with ThreadPoolExecutor(6) as executor:
        for rar in sorted((args.base / "gravacoes").glob("*.rar")):
            converter(rar, args.base, executor)


if __name__ == "__main__":
    main()
