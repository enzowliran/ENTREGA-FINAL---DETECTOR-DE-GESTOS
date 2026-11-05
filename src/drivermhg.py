"""Leitura das anotações e dos frames do DriverMHG."""
import re
import subprocess
from pathlib import Path

import pandas as pd

# Numeração do classIndAllRight.txt, a mesma usada pelo create_json.py oficial. O classIndAll.txt
# traz outra ordem. A conferência visual ainda não confirmou qual está certa (ver notebook 01).
NOMES_GESTOS = {1: "FlickDown", 2: "FlickUp", 3: "SwipeLeft", 4: "SwipeRight", 5: "Tap"}
PESSOAS_TREINO = range(1, 19)
PESSOAS_TESTE = range(19, 26)
FPS = 30


def nome_classe(numero):
    """Converte o número da anotação (1 a 14) em nome da classe e mão."""
    if numero <= 5:
        return NOMES_GESTOS[numero], "direita"
    if numero <= 10:
        return NOMES_GESTOS[numero - 5], "esquerda"
    return {11: ("None", "direita"), 12: ("None", "esquerda"),
            13: ("Other", "direita"), 14: ("Other", "esquerda")}[numero]


def extrair_rar(rar, destino):
    """Extrai um RAR com o tar do Windows, que também entende RAR4."""
    Path(destino).mkdir(parents=True, exist_ok=True)
    subprocess.run([r"C:\Windows\System32\tar.exe", "-xf", str(rar), "-C", str(destino)], check=False)


def ler_listas(pasta_anotacao):
    """Lê as listas oficiais de treino e teste de cada mão em uma única tabela."""
    linhas = []
    for mao in ("left", "right"):
        for subconjunto, arquivo in (("treino", "trainlist"), ("teste", "vallist")):
            for linha in (Path(pasta_anotacao) / f"{arquivo}_all_{mao}.txt").read_text().splitlines():
                partes = linha.split()
                if len(partes) != 4:
                    continue
                gravacao, classe, inicio, fim = partes[0], int(partes[1]), int(partes[2]), int(partes[3])
                achado = re.match(r"Subject(\d+)/Rec(\d+)", gravacao)
                gesto, nome_mao = nome_classe(classe)
                linhas.append({"pessoa": int(achado[1]), "gravacao": int(achado[2]), "classe": classe,
                               "gesto": gesto, "mao": nome_mao, "inicio": inicio, "fim": fim,
                               "n_frames": fim - inicio, "subconjunto": subconjunto})
    return pd.DataFrame(linhas)


def ler_anotacao_gravacao(pasta_gravacao):
    """Lê o annotationall_all.txt de uma gravação, ordenado pelo início."""
    linhas = []
    for linha in (Path(pasta_gravacao) / "annotationall_all.txt").read_text().splitlines():
        partes = linha.split()
        if len(partes) == 4:
            classe = int(partes[1])
            gesto, mao = nome_classe(classe)
            linhas.append({"classe": classe, "gesto": gesto, "mao": mao,
                           "inicio": int(partes[2]), "fim": int(partes[3])})
    return pd.DataFrame(linhas).sort_values("inicio").reset_index(drop=True)
