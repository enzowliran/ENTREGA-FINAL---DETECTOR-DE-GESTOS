"""Transforma os vídeos reais (IPN Hand, Slovo, ASL Citizen) em quadros prontos para o treino dos dedos.

Para cada vídeo, guarda o quadrado central (o mesmo recorte da webcam) reduzido a 128x128, a 12 quadros por segundo,
em JPEG num arquivo único (data/videos_reais/quadros.bin), como o Jester. Em trechos.json fica, para cada trecho,
a classe, a pessoa, a divisão (treino ou validação) e as janelas de 1 s que podem ser usadas.

IPN Hand (divisão oficial por pessoa: 37 no treino, 13 na validação):
  - B0A, apontar com um dedo  -> One Finger, só janelas em que a mão anda até 20% do quadro em 1 s;
  - B0B, apontar com dois     -> Two Fingers, só até 15% (o Jester tem gestos de dois dedos andando, que não podem
                                 virar "dois dedos");
  - D0X, sem gesto            -> No gesture;
  - cliques e abrir duas vezes (G01, G02, G07, G08, G09) -> Doing other things;
  - jogar e zoom (G03-G06, G10, G11) ficam de fora: parecem os deslizar e zoom do Jester.
  O movimento vem das máscaras da mão que acompanham a base (centro da mão em cada quadro).
Slovo: no_event -> No gesture (treino ou validação pela divisão da base); um e dois -> só validação.
ASL Citizen: ONE e TWO -> só validação (a licença não permite redistribuir; nada disso vai para o git).

Uso:
    python src/preparar_videos_reais.py
"""
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "data/videos_reais"
QPS, LADO, JANELA = 12, 128, 12  # 12 quadros por segundo; janelas de 1 s
IPN_CLASSES = {"B0A": ("One Finger", 0.20), "B0B": ("Two Fingers", 0.15), "D0X": ("No gesture", None),
               "G01": ("Doing other things", None), "G02": ("Doing other things", None), "G07": ("Doing other things", None),
               "G08": ("Doing other things", None), "G09": ("Doing other things", None)}


def quadrado(q):
    a, l = q.shape[:2]
    b = min(a, l)
    return cv2.resize(q[(a - b) // 2:(a - b) // 2 + b, (l - b) // 2:(l - b) // 2 + b], (LADO, LADO), interpolation=cv2.INTER_AREA)


def ler_video(caminho, manter=None):
    """Lê o vídeo em sequência e guarda só o quadrado central 128x128 dos quadros pedidos (numeração a partir de 0).

    Guardar o vídeo inteiro em 640x480 gastaria uns 3,5 GB por vídeo do IPN.
    """
    cap = cv2.VideoCapture(str(caminho))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    quadros, i = {}, 0
    while True:
        ok, q = cap.read()
        if not ok:
            break
        if manter is None or i in manter:
            quadros[i] = quadrado(q)
        i += 1
    cap.release()
    return quadros, fps, i


def centro_mao(pasta_mascaras, video, quadro_1):
    """Centro da mão (x, y) em frações do lado do quadrado central; NaN se a mão não aparece."""
    m = cv2.imread(str(pasta_mascaras / video / f"{video}_{quadro_1:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.nan, np.nan
    ys, xs = np.nonzero(m > 127)
    if len(xs) < 50:
        return np.nan, np.nan
    a, l = m.shape
    b = min(a, l)
    return (xs.mean() - (l - b) / 2) / b, (ys.mean() - (a - b) / 2) / b


def processar_ipn(video, trechos_video, pasta, divisao, pessoa):
    fps = 30.0  # todos os vídeos do IPN são 30 quadros por segundo
    por_trecho = [np.arange(int(t["t_start"]), int(t["t_end"]) + 1, fps / QPS).astype(int) for t in trechos_video]  # numeração do IPN começa em 1
    quadros, _, total = ler_video(RAIZ / f"data/IPN/videos/videos/{video}.avi", {int(i) - 1 for ind in por_trecho for i in ind})
    saida = []
    for t, indices in zip(trechos_video, por_trecho):
        classe, limite = IPN_CLASSES[t["label"]]
        indices = indices[indices <= total]
        if len(indices) < JANELA:
            continue
        imagens = [cv2.imencode(".jpg", quadros[i - 1], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes() for i in indices]
        janelas = list(range(len(indices) - JANELA + 1))
        amplitude = None
        if limite is not None:
            c = np.array([centro_mao(pasta, video, int(i)) for i in indices])
            amplitude = []
            for k in janelas:
                j = c[k:k + JANELA]
                # mão ausente ou fora do quadrado em algum quadro: a janela não serve como exemplo do gesto
                fora = np.isnan(j).any() or (j < 0).any() or (j > 1).any()
                amplitude.append(np.inf if fora else float(np.max(j.max(0) - j.min(0))))
            janelas = [k for k in janelas if amplitude[k] <= limite]
        saida.append({"base": "IPN", "video": video, "pessoa": pessoa, "divisao": divisao, "rotulo_original": t["label"], "classe": classe,
                      "imagens": imagens, "janelas": janelas,
                      "amplitude_mediana": float(np.median([a for a in amplitude if np.isfinite(a)])) if amplitude and np.isfinite(amplitude).any() else None})
    return saida


def processar_simples(base, caminho, classe, divisao, pessoa, inicio=None, fim=None):
    lidos, fps, total = ler_video(caminho)  # vídeos curtos: cabem inteiros em 128x128
    quadros = [lidos[i] for i in range(total)]
    if inicio is not None and fim and fim > inicio:
        meio, meia = (inicio + fim) / 2, max((fim - inicio) / 2, fps / 2)  # o sinal, com folga até 1 s
        quadros = quadros[max(0, int(meio - meia)):int(meio + meia) + 1]
    indices = np.arange(0, len(quadros), fps / QPS).astype(int)
    if len(indices) < 4:
        return []
    imagens = [cv2.imencode(".jpg", quadros[i], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes() for i in indices]
    return [{"base": base, "video": Path(caminho).stem, "pessoa": pessoa, "divisao": divisao, "rotulo_original": classe, "classe": classe,
             "imagens": imagens, "janelas": list(range(max(1, len(indices) - JANELA + 1))), "amplitude_mediana": None}]


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    tarefas = []
    meta = {l["Video Name"]: l for l in csv.DictReader(open(RAIZ / "data/IPN/anotacoes/metadata.csv", encoding="utf-8"))}
    anot = {}
    for l in csv.DictReader(open(RAIZ / "data/IPN/anotacoes/Annot_List.txt", encoding="utf-8")):
        if l["label"] in IPN_CLASSES:
            anot.setdefault(l["video"], []).append(l)
    pasta_m = RAIZ / "data/IPN/mascaras/segment"
    for video, trechos in anot.items():
        pessoa = "_".join(video.split("_")[:2])  # 4 vídeos por pessoa: 1CM1_4_R_#229 ... #232
        divisao = "treino" if meta[video]["Set"] == "train" else "validacao"
        tarefas.append((processar_ipn, (video, trechos, pasta_m, divisao, "IPN " + pessoa)))
    for l in csv.DictReader(open(RAIZ / "data/Slovo/annotations.tsv", encoding="utf-8"), delimiter="\t"):
        caminho = RAIZ / f"data/Slovo/videos/{l['attachment_id']}.mp4"
        if not caminho.exists():
            continue
        if l["text"] == "no_event":
            tarefas.append((processar_simples, ("Slovo", caminho, "No gesture", "treino" if l["train"] == "True" else "validacao", "Slovo " + l["user_id"])))
        elif l["text"] in ("один", "два"):
            classe = "One Finger" if l["text"] == "один" else "Two Fingers"
            tarefas.append((processar_simples, ("Slovo", caminho, classe, "validacao_sinais", "Slovo " + l["user_id"], int(l["begin"]), int(l["end"]))))
    for s in ("train", "val", "test"):
        for l in csv.DictReader(open(RAIZ / f"data/ASLCitizen/{s}.csv", encoding="utf-8")):
            if l["Gloss"] in ("ONE", "TWO"):
                caminho = RAIZ / "data/ASLCitizen/videos" / Path(l["Video file"]).name
                if caminho.exists():
                    classe = "One Finger" if l["Gloss"] == "ONE" else "Two Fingers"
                    tarefas.append((processar_simples, ("ASL Citizen", caminho, classe, "validacao_sinais", "ASL " + l["Participant ID"])))

    print(f"{len(tarefas)} vídeos para processar", flush=True)
    trechos, t0 = [], time.time()
    with (SAIDA / "quadros.bin").open("wb") as binario, ThreadPoolExecutor(8) as executor:
        posicao = 0
        for k, resultado in enumerate(executor.map(lambda t: t[0](*t[1]), tarefas), 1):
            for t in resultado:
                deslocamentos = []
                for img in t.pop("imagens"):
                    binario.write(img)
                    deslocamentos.append([posicao, len(img)])
                    posicao += len(img)
                t["quadros"] = deslocamentos
                trechos.append(t)
            if k % 25 == 0:
                print(f"  {k}/{len(tarefas)} vídeos, {posicao / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)
    (SAIDA / "trechos.json").write_text(json.dumps(trechos), encoding="utf-8")

    # Resumo: janelas de 1 s utilizáveis por base, divisão e classe.
    resumo = {}
    for t in trechos:
        chave = f"{t['base']} · {t['divisao']} · {t['classe']}"
        r = resumo.setdefault(chave, {"trechos": 0, "trechos_com_janela": 0, "janelas": 0, "pessoas": set()})
        r["trechos"] += 1
        r["trechos_com_janela"] += bool(t["janelas"])
        r["janelas"] += len(t["janelas"])
        r["pessoas"].add(t["pessoa"])
    for chave in sorted(resumo):
        r = resumo[chave]
        print(f"{chave}: {r['trechos']} trechos ({r['trechos_com_janela']} com janela útil), {r['janelas']} janelas de 1 s, {len(r['pessoas'])} pessoas")
    (SAIDA / "resumo.json").write_text(json.dumps({k: {**v, "pessoas": len(v["pessoas"])} for k, v in resumo.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Pronto em {(time.time() - t0) / 60:.1f} min: {SAIDA}")


if __name__ == "__main__":
    main()
