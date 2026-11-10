"""Diagnóstico: o modelo enxerga um e dois dedos em vídeo real se a mão aparecer maior?

Roda o modelo atual (sem treinar nada) nos vídeos de "um" e "dois" do Slovo e do ASL Citizen, com vários recortes:
o quadrado central inteiro (o que a webcam usa hoje) e zooms de 75%, 50% e 35% numa grade 3x3.

Para cada recorte anota a probabilidade do gesto certo e a posição dele no ranking das 32 saídas. "Melhor recorte"
escolhe, por vídeo, o recorte com maior probabilidade do gesto certo; como escolher o melhor de 28 sempre ajuda um
pouco, o controle faz a mesma escolha para o gesto errado (dois dedos num vídeo de um, e vice-versa).

Grava só números em resultados/dedos/diagnostico_escala.json (nenhum quadro: as licenças não permitem publicar).

Uso:
    python src/diagnostico_escala.py
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import modelos as M

RAIZ = Path(__file__).resolve().parent.parent
QPS, N, LADO = 12, 16, 96
ESCALAS = [1.0, 0.75, 0.5, 0.35]
ALVO = {"один": "One Finger", "два": "Two Fingers", "ONE": "One Finger", "TWO": "Two Fingers"}


def ler(caminho, inicio=None, fim=None):
    """Quadros BGR a 12 por segundo; com início e fim (em quadros originais), só o trecho do sinal, com folga até 1 s."""
    cap = cv2.VideoCapture(str(caminho))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    quadros = []
    while True:
        ok, q = cap.read()
        if not ok:
            break
        quadros.append(q)
    cap.release()
    if inicio is not None and fim and fim > inicio:
        meio, meia = (inicio + fim) / 2, max((fim - inicio) / 2, fps / 2)
        quadros = quadros[max(0, int(meio - meia)):int(meio + meia) + 1]
    indices = np.arange(0, len(quadros), fps / QPS).astype(int)
    return [quadros[i] for i in indices if i < len(quadros)]


def recortes(altura, largura):
    """(nome, y0, x0, lado) de cada recorte: o quadrado central e zooms numa grade 3x3 dentro dele."""
    base = min(altura, largura)
    y0, x0 = (altura - base) // 2, (largura - base) // 2
    saida = [("centro 100%", y0, x0, base)]
    for e in ESCALAS[1:]:
        lado = int(base * e)
        for i, fy in enumerate((0, 0.5, 1)):
            for j, fx in enumerate((0, 0.5, 1)):
                saida.append((f"zoom {int(e * 100)}% ({i},{j})", y0 + int((base - lado) * fy), x0 + int((base - lado) * fx), lado))
    return saida


def clipe(quadros, y0, x0, lado):
    idx = np.round(np.linspace(0, len(quadros) - 1, N)).astype(int)
    c = np.stack([cv2.resize(quadros[i][y0:y0 + lado, x0:x0 + lado], (LADO, LADO), interpolation=cv2.INTER_AREA)[..., ::-1].transpose(2, 0, 1) for i in idx])
    return c.astype(np.float32)


def main():
    estado = torch.load(RAIZ / "resultados/jester/cnn_3d_mais_gestos/melhor.pt", map_location="cuda", weights_only=False)
    nomes = list(estado["nomes"])
    modelo = M.criar_modelo(estado["arquitetura"], len(nomes)).cuda().eval()
    modelo.load_state_dict(estado["modelo"])

    videos = []
    for l in csv.DictReader(open(RAIZ / "data/Slovo/annotations.tsv", encoding="utf-8"), delimiter="\t"):
        if l["text"] in ("один", "два"):
            videos.append(("Slovo", RAIZ / f"data/Slovo/videos/{l['attachment_id']}.mp4", ALVO[l["text"]], int(l["begin"]), int(l["end"])))
    for s in ("train", "val", "test"):
        for l in csv.DictReader(open(RAIZ / f"data/ASLCitizen/{s}.csv", encoding="utf-8")):
            if l["Gloss"] in ("ONE", "TWO"):
                videos.append(("ASL Citizen", RAIZ / "data/ASLCitizen/videos" / Path(l["Video file"]).name, ALVO[l["Gloss"]], None, None))

    linhas = []
    for base, caminho, alvo, ini, fim in videos:
        if not caminho.exists():
            continue
        quadros = ler(caminho, ini, fim)
        if len(quadros) < 4:
            continue
        rs = recortes(*quadros[0].shape[:2])
        x = torch.from_numpy(np.stack([clipe(quadros, y, x0, l) for _, y, x0, l in rs])).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = modelo((x - 127.5) / 127.5).float().softmax(1).cpu().numpy()
        a, outro = nomes.index(alvo), nomes.index("Two Fingers" if alvo == "One Finger" else "One Finger")
        posicao = (p > p[:, [a]]).sum(1) + 1
        for k, (nome, *_rest) in enumerate(rs):
            linhas.append({"base": base, "video": caminho.stem, "alvo": alvo, "recorte": nome, "escala": nome.split()[1],
                           "p_alvo": float(p[k, a]), "p_outro_dedo": float(p[k, outro]), "posicao": int(posicao[k]), "top1": nomes[int(p[k].argmax())]})

    def resumo(sel):
        if not sel:
            return None
        return {"videos": len(sel), "p_alvo_media": round(float(np.mean([l["p_alvo"] for l in sel])), 3),
                "top1": round(float(np.mean([l["posicao"] == 1 for l in sel])), 3), "top5": round(float(np.mean([l["posicao"] <= 5 for l in sel])), 3),
                "p_alvo_mediana": round(float(np.median([l["p_alvo"] for l in sel])), 3)}

    saida = {"modelo": "resultados/jester/cnn_3d_mais_gestos/melhor.pt", "por_base": {}}
    for base in ("Slovo", "ASL Citizen", "todas"):
        for alvo in ("One Finger", "Two Fingers", "ambos"):
            sel = [l for l in linhas if (base == "todas" or l["base"] == base) and (alvo == "ambos" or l["alvo"] == alvo)]
            if not sel:
                continue
            r = {}
            for esc in ("100%", "75%", "50%", "35%"):
                # por vídeo, o melhor dos recortes desta escala (na escala 100% só existe um)
                por_video = {}
                for l in sel:
                    if l["escala"] == esc and (l["video"] not in por_video or l["p_alvo"] > por_video[l["video"]]["p_alvo"]):
                        por_video[l["video"]] = l
                r[f"melhor recorte {esc}"] = resumo(list(por_video.values()))
            melhor, controle = {}, {}
            for l in sel:
                if l["video"] not in melhor or l["p_alvo"] > melhor[l["video"]]["p_alvo"]:
                    melhor[l["video"]] = l
                if l["video"] not in controle or l["p_outro_dedo"] > controle[l["video"]]:
                    controle[l["video"]] = l["p_outro_dedo"]
            r["melhor de todos os 28"] = resumo(list(melhor.values()))
            r["controle: melhor dos 28 para o dedo errado (p media)"] = round(float(np.mean(list(controle.values()))), 3)
            r["mais votado no centro"] = {k: v for k, v in sorted(
                {n: sum(1 for l in sel if l["escala"] == "100%" and l["top1"] == n) for n in {l["top1"] for l in sel}}.items(), key=lambda x: -x[1])[:4] if v}
            saida["por_base"][f"{base} · {alvo}"] = r
    pasta = RAIZ / "resultados/dedos"
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "diagnostico_escala.json").write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    for chave, r in saida["por_base"].items():
        print(f"\n== {chave}")
        for k, v in r.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
