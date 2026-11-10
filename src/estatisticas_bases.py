"""Calcula as estatísticas das três bases de dados usadas, para a apresentação e o paper.

Lê os arquivos locais (rótulos do Jester, índice e anotações do HaGRIDv2, tabelas do DriverMHG) e grava
resultados/estatisticas_bases.json. Assim a apresentação não depende dos dados brutos, que ficam fora do git.

Uso:
    python src/estatisticas_bases.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import jester as J

RAIZ = Path(".")


def tamanho_gb(pasta):
    return round(sum(f.stat().st_size for f in Path(pasta).rglob("*") if f.is_file()) / 1e9, 1)


def jester():
    nomes, treino, validacao, teste = J.ler_rotulos(Path("data/Jester/rotulos/jester_labels"))
    base = json.loads(Path("resultados/jester/dados_jester.json").read_text(encoding="utf-8"))
    por_classe = Counter(nomes[c] for c in treino.values())
    indice = np.fromfile("data/Jester/jester_indice.bin", dtype=J.TIPO_INDICE)
    quadros_por_video = np.bincount(indice["video"])
    quadros_por_video = quadros_por_video[quadros_por_video > 0]
    base.update({
        "treino_por_classe": dict(sorted(por_classe.items(), key=lambda x: -x[1])),
        "quadros_por_video_media": round(float(quadros_por_video.mean()), 1),
        "duracao_media_s": round(float(quadros_por_video.mean()) / 12, 1),
        "fps": 12, "altura_px": 100,
    })
    return base


def hagrid():
    indice = json.loads(Path("data/HaGRID/indice_imagens.json").read_text(encoding="utf-8"))
    # o zip traz uma pasta .ipynb_checkpoints com 9 cópias, que não é classe
    classes_zip = Counter(Path(i[0]).parent.name for i in indice
                          if i[0].lower().endswith(".jpg") and not Path(i[0]).parent.name.startswith("."))
    anot = json.loads(Path("data/HaGRID/anotacoes.json").read_text(encoding="utf-8"))
    usadas, pessoas = {}, defaultdict(set)
    for divisao in ("train", "val"):
        for pasta in sorted((Path("data/HaGRID") / divisao).iterdir()):
            arquivos = list(pasta.glob("*.jpg"))
            usadas.setdefault(pasta.name, {})[divisao] = len(arquivos)
            for a in arquivos:
                if a.stem in anot:
                    pessoas[divisao].add(anot[a.stem]["user_id"])
    return {
        "imagens_no_arquivo": sum(classes_zip.values()), "classes_no_arquivo": len(classes_zip),
        "gb_arquivo_completo": 128.3, "gb_baixados": round(tamanho_gb("data/HaGRID/train") + tamanho_gb("data/HaGRID/val"), 1),
        "lado_min_px": 512, "usadas": usadas,
        "fotos_treino": sum(v.get("train", 0) for v in usadas.values()), "fotos_validacao": sum(v.get("val", 0) for v in usadas.values()),
        "pessoas_treino": len(pessoas["train"]), "pessoas_validacao": len(pessoas["val"]),
        "pessoas_em_comum": len(pessoas["train"] & pessoas["val"]),
    }


def drivermhg():
    classes = list(csv.DictReader(open("resultados/drivermhg_por_classe.csv", encoding="utf-8")))
    pessoas = list(csv.DictReader(open("resultados/drivermhg_por_pessoa.csv", encoding="utf-8")))
    duracao = list(csv.DictReader(open("resultados/drivermhg_duracao.csv", encoding="utf-8")))
    progresso = json.loads(Path("data/DriverMHG/progresso.json").read_text(encoding="utf-8"))
    por_gesto = defaultdict(int)
    for c in classes:
        por_gesto[c["gesto"]] += int(c["total"])
    return {
        "pessoas": len(pessoas), "segmentos": sum(int(p["segmentos"]) for p in pessoas),
        "gravacoes": sum(int(p["gravacoes"]) for p in pessoas),
        "pessoas_treino": sum(1 for p in pessoas if p["subconjunto"] == "treino"), "pessoas_teste": sum(1 for p in pessoas if p["subconjunto"] == "teste"),
        "por_gesto": dict(por_gesto), "duracao_media_s": round(sum(float(d["media_s"]) * int(d["count"]) for d in duracao) / sum(int(d["count"]) for d in duracao), 2),
        "modalidades": ["RGB", "infravermelho", "profundidade"], "resolucao": "320×240", "fps": 30,
        "gravacoes_baixadas": len(progresso["concluidas"]), "gb_baixados": tamanho_gb("data/DriverMHG"),
    }


def main():
    saida = {"jester": jester(), "hagrid": hagrid(), "drivermhg": drivermhg()}
    Path("resultados/estatisticas_bases.json").write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    j, h, d = saida["jester"], saida["hagrid"], saida["drivermhg"]
    print(f"Jester: {j['videos_total']} vídeos, {j['quadros_por_video_media']} quadros por vídeo ({j['duracao_media_s']} s)")
    print(f"HaGRID: {h['imagens_no_arquivo']} imagens em {h['classes_no_arquivo']} classes; usamos {h['fotos_treino']} + {h['fotos_validacao']} fotos, "
          f"{h['pessoas_treino']} pessoas no treino e {h['pessoas_validacao']} na validação, {h['pessoas_em_comum']} em comum; {h['gb_baixados']} GB")
    print(f"DriverMHG: {d['pessoas']} pessoas, {d['segmentos']} segmentos, {d['gravacoes']} gravações; baixadas {d['gravacoes_baixadas']}, {d['gb_baixados']} GB")


if __name__ == "__main__":
    main()
