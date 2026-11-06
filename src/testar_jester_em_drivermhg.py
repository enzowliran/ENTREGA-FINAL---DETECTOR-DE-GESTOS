"""Testa o modelo do Jester, sem ajuste nenhum, nos segmentos já preparados do DriverMHG.

O modelo só conhece as classes do Jester, então cada gesto é comparado com os equivalentes mais
próximos. Toque não tem equivalente. Sem frames na saída, por causa da licença do DriverMHG.

Uso:
    python src/testar_jester_em_drivermhg.py
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import drivermhg as D
import modelos as M

EQUIVALENTES = {
    "SwipeLeft": ["Swiping Left", "Sliding Two Fingers Left"],
    "SwipeRight": ["Swiping Right", "Sliding Two Fingers Right"],
    "FlickUp": ["Swiping Up", "Sliding Two Fingers Up"],
    "FlickDown": ["Swiping Down", "Sliding Two Fingers Down"],
    "None": ["No gesture", "Doing other things"],
    "Other": ["No gesture", "Doing other things"],
}
SEM_GESTO = ("No gesture", "Doing other things")


def clipe(video, inicio, fim, modo):
    """16 frames do segmento, com recorte central quadrado, como na validação do Jester."""
    indices = np.minimum(np.linspace(inicio, max(fim, inicio + 1), 16).astype(int), len(video) - 1)
    q = np.asarray(video[indices], dtype=np.float32)
    if q.ndim == 3:
        q = np.repeat(q[..., None], 3, axis=-1)
    x0 = (q.shape[2] - q.shape[1]) // 2
    q = q[:, :, x0:x0 + q.shape[1]]
    if modo == "girado":
        q = q[:, ::-1, ::-1].copy()
    return q.transpose(0, 3, 1, 2)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--modelo", default="resultados/jester/cnn_3d/melhor.pt")
    a.add_argument("--prep", default="data/DriverMHG/prep")
    a.add_argument("--saida", default="resultados/drivermhg/jester_sem_ajuste.csv")
    args = a.parse_args()
    estado = torch.load(args.modelo, map_location="cuda", weights_only=False)
    nomes = estado["nomes"]
    modelo = M.criar_modelo(estado["arquitetura"], len(nomes)).cuda().eval()
    modelo.load_state_dict(estado["modelo"])
    indices_gesto = [i for i, n in enumerate(nomes) if n not in SEM_GESTO]

    segmentos = []
    arquivos = sorted(Path(args.prep).glob("*_ir.npy"))
    for ir in arquivos:
        base = ir.name[:-len("_ir.npy")]
        for linha in (Path(args.prep) / f"{base}_anotacao.txt").read_text().splitlines():
            p = linha.split()
            if len(p) == 4:
                gesto, mao = D.nome_classe(int(p[1]))
                segmentos.append((base, gesto, mao, int(p[2]), int(p[3])))
    print(f"{len(arquivos)} gravações, {len(segmentos)} segmentos", flush=True)

    linhas, resumo = [], {}
    for fonte, modo in (("rgb", "normal"), ("rgb", "girado"), ("ir", "normal")):
        contas = collections.defaultdict(collections.Counter)
        ok, total, disparos_neg, neg, disparos_pos, pos = 0, 0, 0, 0, 0, 0
        abertos = {}
        lotes, info = [], []

        def processar():
            nonlocal ok, total, disparos_neg, neg, disparos_pos, pos
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = (torch.from_numpy(np.stack(lotes)).cuda() - 127.5) / 127.5
                prob = modelo(x).float().softmax(1).cpu().numpy()
            for (base, gesto, mao, ini, fim), q in zip(info, prob):
                top = nomes[int(q.argmax())]
                contas[gesto][top] += 1
                confianca_gesto = float(q[indices_gesto].max())
                dispara = confianca_gesto >= 0.6
                if gesto in EQUIVALENTES:
                    total += 1
                    ok += top in EQUIVALENTES[gesto]
                if gesto in ("None", "Other"):
                    neg += 1; disparos_neg += dispara
                else:
                    pos += 1; disparos_pos += dispara
                linhas.append({"fonte": fonte, "modo": modo, "gravacao": base, "gesto": gesto, "mao": mao, "inicio": ini, "fim": fim,
                               "predito": top, "confianca": round(float(q.max()), 4), "conf_gesto": round(confianca_gesto, 4)})
            lotes.clear(); info.clear()

        for s in segmentos:
            base = s[0]
            if base not in abertos:
                abertos[base] = np.load(Path(args.prep) / f"{base}_{fonte}.npy", mmap_mode="r")
            lotes.append(clipe(abertos[base], s[3], s[4], modo)); info.append(s)
            if len(lotes) == 48:
                processar()
        if lotes:
            processar()
        chave = f"{fonte}_{modo}"
        resumo[chave] = {"acerto_equivalentes": ok / total, "n_equivalentes": total, "dispara_em_sem_gesto": disparos_neg / neg,
                         "n_sem_gesto": neg, "dispara_em_gesto": disparos_pos / pos, "n_gesto": pos,
                         "mais_previstos": {g: c.most_common(3) for g, c in sorted(contas.items())}}
        print(f"\n=== {chave}: acerto com equivalentes {ok}/{total} = {ok / total:.1%}; "
              f"dispara em None/Other {disparos_neg / neg:.1%}; dispara em gestos {disparos_pos / pos:.1%} (conf. >= 0,6)", flush=True)
        for g, c in sorted(contas.items()):
            print(f"  {g:11s} n={sum(c.values()):3d} -> " + ", ".join(f"{n} {v}" for n, v in c.most_common(3)), flush=True)

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open("w", newline="", encoding="utf-8") as f:
        e = csv.DictWriter(f, fieldnames=list(linhas[0]))
        e.writeheader(); e.writerows(linhas)
    saida.with_suffix(".json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
